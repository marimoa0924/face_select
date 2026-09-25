"""명령행 진입점.

    python -m face_select enroll                     # reference/ 사진으로 내 얼굴 등록
    python -m face_select scan [--folder ID]         # 드라이브 사진 스캔(재개 가능)
    python -m face_select report [--threshold 0.4]   # 매칭 결과 CSV 출력
    python -m face_select export [--threshold 0.4]   # 드라이브에 '내 사진' 폴더 + 바로가기 생성
"""

from __future__ import annotations

import argparse
import csv
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from .store import Store

REF_PATH = "reference_embeddings.npy"


def cmd_enroll(args):
    from .faces import FaceEngine, ReferenceSet

    engine = FaceEngine(use_gpu=args.gpu)
    refs = ReferenceSet.from_folder(engine, args.reference_dir)
    refs.save(REF_PATH)
    print(f"기준 얼굴 {len(refs.embeddings)}개 등록 → {REF_PATH}")


def cmd_scan(args):
    from .drive import DriveClient, authenticate
    from .faces import FaceEngine, decode_image

    store = Store(args.db)
    engine = FaceEngine(use_gpu=args.gpu, min_face_px=args.min_face)
    drive = DriveClient(authenticate(args.credentials))

    print("드라이브에서 사진 목록을 가져오는 중...")
    photos = [p for p in drive.list_photos(args.folder) if not store.is_done(p.id, p.md5)]
    if args.limit:
        photos = photos[: args.limit]
    print(f"새로 처리할 사진: {len(photos)}장")

    # 다운로드(네트워크 I/O)는 스레드 풀, 추론(CPU/GPU)은 메인 스레드에서 처리.
    # 각 스레드가 자체 HTTP 세션을 쓰도록 스레드 로컬 클라이언트를 사용한다.
    local = threading.local()

    def fetch(photo):
        if not hasattr(local, "drive"):
            local.drive = DriveClient(drive.session.credentials)
        try:
            return photo, local.drive.fetch_image_bytes(photo, args.size), None
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
    store = Store(args.db)
    rows = sorted(_matches(store, args.threshold - args.review_margin), key=lambda r: -r[3])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file_id", "name", "taken_time", "score", "verdict", "url"])
        for pid, name, taken, score in rows:
            verdict = "match" if score >= args.threshold else "review"
            w.writerow([pid, name, taken or "", f"{score:.3f}", verdict,
                        f"https://drive.google.com/file/d/{pid}/view"])
    n_match = sum(r[3] >= args.threshold for r in rows)
    print(f"확실: {n_match}장, 검토 필요: {len(rows) - n_match}장 → {args.out}")


def cmd_export(args):
    from .drive import DriveClient, authenticate

    store = Store(args.db)
    drive = DriveClient(authenticate(args.credentials))
    folder_id = drive.ensure_folder(args.folder_name)
    rows = list(_matches(store, args.threshold))
    added = 0
    for pid, name, _, _ in tqdm(rows, unit="장"):
        if store.is_exported(pid, folder_id):
            continue
        store.mark_exported(pid, folder_id, drive.add_shortcut(pid, name, folder_id))
        added += 1
    print(f"'{args.folder_name}' 폴더에 바로가기 {added}개 추가 (총 매칭 {len(rows)}장)")


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
    s.add_argument("--folder", help="이 폴더 ID 하위만 스캔 (생략 시 드라이브 전체)")
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

    s = sub.add_parser("export", help="드라이브에 결과 폴더를 만들고 매칭 사진 바로가기 추가")
    s.add_argument("--threshold", type=float, default=0.40)
    s.add_argument("--folder-name", default="내 얼굴 사진 (face_select)")
    s.set_defaults(func=cmd_export)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
