"""명령행 진입점.

    python -m face_select enroll                     # reference/ 사진으로 내 얼굴 등록
    python -m face_select scan [--folder URL|ID]     # 드라이브 사진 스캔(재개 가능)
    python -m face_select scan --local DIR           # PC에 내려받은 사진 폴더 스캔
    python -m face_select report [--threshold 0.4]   # 매칭 결과 CSV 출력
    python -m face_select export [--threshold 0.4]   # 드라이브 바로가기 생성 / 로컬 사진은 복사
"""

from __future__ import annotations

import argparse
import csv
import queue
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from .store import Store

REF_PATH = "reference_embeddings.npy"


def folder_id(value: str) -> str:
    """폴더 ID 또는 드라이브 폴더 URL(모바일/공유 링크 포함)에서 ID를 뽑는다."""
    m = re.search(r"/folders/([\w-]+)", value) or re.search(r"[?&]id=([\w-]+)", value)
    return m.group(1) if m else value


def cmd_enroll(args):
    from .faces import FaceEngine, ReferenceSet

    engine = FaceEngine(use_gpu=args.gpu)
    refs = ReferenceSet.from_folder(engine, args.reference_dir)
    refs.save(REF_PATH)
    print(f"기준 얼굴 {len(refs.embeddings)}개 등록 → {REF_PATH}")


def _list_drive_photos(args):
    """드라이브 사진 목록과 다운로드 함수. 폴더 접근/목록 문제를 진단해 알려준다."""
    from googleapiclient.errors import HttpError

    from .drive import DriveClient, authenticate

    drive = DriveClient(authenticate(args.credentials))
    account = drive.whoami()
    print(f"로그인한 계정: {account}")
    if args.folder:
        try:
            info = drive.folder_info(args.folder)
        except HttpError as e:
            if e.resp.status in (403, 404):
                raise SystemExit(
                    f"이 계정({account})으로는 폴더({args.folder})에 접근할 수 없습니다.\n"
                    "  - 폴더를 볼 수 있는 다른 계정으로 로그인하려면 token.json 을 지우고 다시 실행하세요.\n"
                    "  - 또는 폴더를 PC로 내려받아 scan --local <폴더경로> 로 분석할 수 있습니다."
                ) from e
            raise
        owners = ", ".join(o["emailAddress"] for o in info.get("owners", [])) or "알 수 없음"
        print(f"대상 폴더: {info['name']} (소유자: {owners})")

    print("드라이브에서 사진 목록을 가져오는 중...")
    photos = list(drive.list_photos(args.folder))
    if args.folder:
        w = drive.last_walk
        print(f"탐색한 폴더: {w['folders']}개 (따라간 바로가기: {w['shortcuts']}개)")
    if args.folder and not photos:
        raise SystemExit(
            "폴더에는 접근할 수 있지만 하위 폴더/바로가기를 모두 살펴봐도 사진이 없습니다.\n"
            "  '링크가 있는 모든 사용자'로만 공유된 폴더는 드라이브 API가 안의 파일 목록을 돌려주지 않을 수 있습니다.\n"
            "  해결 방법 (둘 중 하나):\n"
            f"   1) 폴더 소유자에게 내 계정({account})을 직접 공유 대상으로 추가해 달라고 요청\n"
            "   2) 드라이브 웹에서 폴더를 '다운로드'해 압축을 푼 뒤:\n"
            "      python -m face_select scan --local \"압축 푼 폴더 경로\""
        )

    local = threading.local()

    def fetch_bytes(photo):
        # 스레드마다 자체 HTTP 세션을 쓰도록 스레드 로컬 클라이언트 사용
        if not hasattr(local, "drive"):
            local.drive = DriveClient(drive.session.credentials)
        return local.drive.fetch_image_bytes(photo, args.size)

    return photos, fetch_bytes


def _list_local_photos(args):
    from .local import list_local_photos

    root = Path(args.local)
    if not root.is_dir():
        raise SystemExit(f"폴더를 찾을 수 없습니다: {root}")
    print(f"로컬 폴더: {root.resolve()}")
    return list(list_local_photos(root)), lambda photo: photo.path.read_bytes()


def cmd_scan(args):
    from .faces import FaceEngine, decode_image

    store = Store(args.db)
    if args.local:
        all_photos, fetch_bytes = _list_local_photos(args)
    else:
        all_photos, fetch_bytes = _list_drive_photos(args)

    photos = [p for p in all_photos if not store.is_done(p.id, p.md5)]
    print(f"찾은 사진: {len(all_photos)}장 (이미 분석: {len(all_photos) - len(photos)}장, 새로 분석: {len(photos)}장)")
    if args.limit:
        photos = photos[: args.limit]
    if not photos:
        return
    engine = FaceEngine(use_gpu=args.gpu, min_face_px=args.min_face)

    # 다운로드/파일 읽기(I/O)는 스레드 풀, 추론(CPU/GPU)은 메인 스레드에서 처리.
    def fetch(photo):
        try:
            return photo, fetch_bytes(photo), None
        except Exception as e:  # noqa: BLE001
            return photo, None, e

    results: queue.Queue = queue.Queue(maxsize=args.workers * 4)

    def producer():
        with ThreadPoolExecutor(args.workers) as pool:
            for item in pool.map(fetch, photos):
                results.put(item)
        results.put(None)

    threading.Thread(target=producer, daemon=True).start()
    with tqdm(total=len(photos), unit="장") as bar:
        while (item := results.get()) is not None:
            photo, data, err = item
            try:
                if err:
                    raise err
                faces = engine.detect(decode_image(data, args.size))
                store.save_result(photo, faces)
            except Exception as e:  # noqa: BLE001
                store.save_error(photo, repr(e))
            bar.update()
    print("스캔 완료:", store.stats())


def _matches(store: Store, threshold: float):
    from .faces import ReferenceSet

    refs = ReferenceSet.load(REF_PATH)
    for pid, name, taken, embs in store.iter_photo_embeddings():
        score = max(refs.score(e) for e in embs)
        if score >= threshold:
            yield pid, name, taken, score


def cmd_report(args):
    from .local import is_local, local_path

    store = Store(args.db)
    rows = sorted(_matches(store, args.threshold - args.review_margin), key=lambda r: -r[3])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file_id", "name", "taken_time", "score", "verdict", "url"])
        for pid, name, taken, score in rows:
            verdict = "match" if score >= args.threshold else "review"
            url = str(local_path(pid)) if is_local(pid) else f"https://drive.google.com/file/d/{pid}/view"
            w.writerow([pid, name, taken or "", f"{score:.3f}", verdict, url])
    n_match = sum(r[3] >= args.threshold for r in rows)
    print(f"확실: {n_match}장, 검토 필요: {len(rows) - n_match}장 → {args.out}")


def cmd_export(args):
    import shutil

    from .local import is_local, local_path

    store = Store(args.db)
    rows = list(_matches(store, args.threshold))
    drive_rows = [r for r in rows if not is_local(r[0])]
    local_rows = [r for r in rows if is_local(r[0])]

    if drive_rows and not args.skip_drive:
        from .drive import DriveClient, authenticate

        drive = DriveClient(authenticate(args.credentials))
        folder_id = drive.ensure_folder(args.folder_name)
        added = 0
        for pid, name, _, _ in tqdm(drive_rows, unit="장"):
            if store.is_exported(pid, folder_id):
                continue
            store.mark_exported(pid, folder_id, drive.add_shortcut(pid, name, folder_id))
            added += 1
        print(f"드라이브 '{args.folder_name}' 폴더에 바로가기 {added}개 추가 (드라이브 매칭 {len(drive_rows)}장)")

    if local_rows:
        dest = Path(args.copy_to)
        dest.mkdir(parents=True, exist_ok=True)
        copied = 0
        for pid, _, _, _ in tqdm(local_rows, unit="장"):
            src = local_path(pid)
            target, n = dest / src.name, 1
            # 하위 폴더가 달라 이름만 같은 다른 사진은 덮어쓰지 않고 번호를 붙인다
            while target.exists() and target.stat().st_size != src.stat().st_size:
                target, n = dest / f"{src.stem}_{n}{src.suffix}", n + 1
            if target.exists():
                continue
            shutil.copy2(src, target)
            copied += 1
        print(f"로컬 사진 {copied}장 복사 → {dest.resolve()} (로컬 매칭 {len(local_rows)}장)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="face_select", description="구글 드라이브에서 내 얼굴이 있는 사진 추리기")
    p.add_argument("--db", default="cache.sqlite3")
    p.add_argument("--credentials", default="credentials.json")
    p.add_argument("--gpu", action="store_true", help="CUDA 사용 (onnxruntime-gpu 필요)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("enroll", help="reference/ 폴더의 내 사진으로 기준 얼굴 등록")
    s.add_argument("--reference-dir", default="reference")
    s.set_defaults(func=cmd_enroll)

    s = sub.add_parser("scan", help="드라이브 사진의 얼굴을 검출해 캐시에 저장")
    s.add_argument("--folder", type=folder_id,
                   help="이 폴더(ID 또는 공유 링크) 하위만 스캔 (생략 시 드라이브 전체)")
    s.add_argument("--local", help="드라이브 대신 PC의 이 폴더(하위 폴더 포함)를 스캔")
    s.add_argument("--size", type=int, default=1600, help="분석 해상도(긴 변 px). 0이면 원본 다운로드")
    s.add_argument("--min-face", type=int, default=40, help="이보다 작은 얼굴(px)은 무시")
    s.add_argument("--workers", type=int, default=8, help="동시 다운로드 수")
    s.add_argument("--limit", type=int, help="시험용: 최대 N장만 처리")
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("report", help="매칭 결과를 CSV로 저장")
    s.add_argument("--threshold", type=float, default=0.40)
    s.add_argument("--review-margin", type=float, default=0.08, help="threshold 아래 이 폭까지는 '검토' 로 표시")
    s.add_argument("--out", default="output/my_photos.csv")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("export", help="매칭 사진을 드라이브 바로가기(드라이브 사진) 또는 복사(로컬 사진)로 모으기")
    s.add_argument("--threshold", type=float, default=0.40)
    s.add_argument("--folder-name", default="내 얼굴 사진 (face_select)")
    s.add_argument("--copy-to", default="output/my_photos", help="로컬 스캔 사진 중 매칭된 사진을 복사할 폴더")
    s.add_argument("--skip-drive", action="store_true", help="드라이브 바로가기는 만들지 않음")
    s.set_defaults(func=cmd_export)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
