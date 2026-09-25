"""Google Drive 접근: 인증, 사진 목록 조회, 이미지 다운로드, 결과 폴더 생성."""

from __future__ import annotations

import glob
import io
import os
import time
from dataclasses import dataclass
from typing import Iterator

from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# 읽기 전용 + 이 앱이 만든 파일(결과 폴더/바로가기)만 쓰기
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
]
FOLDER_MIME = "application/vnd.google-apps.folder"
SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
LIST_FIELDS = (
    "nextPageToken, files(id, name, mimeType, md5Checksum, size, "
    "thumbnailLink, createdTime, parents, imageMediaMetadata(width, height, time))"
)


@dataclass
class DrivePhoto:
    id: str
    name: str
    mime_type: str
    md5: str | None
    thumbnail_link: str | None
    created_time: str | None
    taken_time: str | None


def _missing_credentials_message(path: str) -> str:
    here = os.path.abspath(os.path.dirname(path) or ".")
    lines = [f"'{path}' 파일을 찾을 수 없습니다.", f"  찾은 위치: {here}"]
    # 흔한 실수: 확장자 숨김으로 생긴 이중 확장자, 이름을 안 바꾼 원본, 다운로드 폴더에 남아 있는 파일
    patterns = ["credentials.json.*", "credentials*.txt", "client_secret*.json",
                os.path.join(os.path.expanduser("~"), "Downloads", "client_secret*.json")]
    found = sorted({os.path.abspath(f) for pat in patterns for f in glob.glob(pat)})
    if found:
        lines.append("  비슷한 파일을 찾았습니다. 이 폴더로 옮기고 이름을 credentials.json 으로 바꿔 주세요:")
        lines += [f"    - {f}" for f in found]
    else:
        lines.append("  구글 클라우드 콘솔에서 받은 OAuth 클라이언트 JSON을 이 폴더에 credentials.json 으로 저장해 주세요.")
    return "\n".join(lines)


def authenticate(credentials_path: str = "credentials.json", token_path: str = "token.json") -> Credentials:
    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                raise SystemExit(_missing_credentials_message(credentials_path))
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as f:
            f.write(creds.to_json())
    return creds


class DriveClient:
    def __init__(self, creds: Credentials):
        self.service = build("drive", "v3", credentials=creds, cache_discovery=False)
        self.session = AuthorizedSession(creds)

    def whoami(self) -> str:
        return self.service.about().get(fields="user(emailAddress)").execute()["user"]["emailAddress"]

    def folder_info(self, folder_id: str) -> dict:
        """폴더 메타데이터. 접근할 수 없으면 HttpError(404/403)."""
        return self.service.files().get(
            fileId=folder_id, fields="id, name, mimeType, owners(emailAddress)", supportsAllDrives=True
        ).execute()

    def _folder_ids_recursive(self, root_id: str) -> list[str]:
        ids, queue = [root_id], [root_id]
        while queue:
            parent = queue.pop()
            q = f"'{parent}' in parents and mimeType = '{FOLDER_MIME}' and trashed = false"
            for f in self._paged_list(q, "nextPageToken, files(id)"):
                ids.append(f["id"])
                queue.append(f["id"])
        return ids

    def _paged_list(self, q: str, fields: str) -> Iterator[dict]:
        token = None
        while True:
            resp = self._retry(
                lambda: self.service.files()
                .list(q=q, fields=fields, pageSize=1000, pageToken=token,
                      supportsAllDrives=True, includeItemsFromAllDrives=True)
                .execute()
            )
            yield from resp.get("files", [])
            token = resp.get("nextPageToken")
            if not token:
                return

    def list_photos(self, folder_id: str | None = None) -> Iterator[DrivePhoto]:
        """이미지 파일을 순회한다. folder_id를 주면 그 하위 폴더까지 재귀적으로 검색."""
        base = "mimeType contains 'image/' and trashed = false"
        if folder_id:
            folders = self._folder_ids_recursive(folder_id)
            # 쿼리 길이 제한을 피하기 위해 폴더를 묶어서 조회
            chunks = [folders[i:i + 30] for i in range(0, len(folders), 30)]
            queries = [f"{base} and (" + " or ".join(f"'{p}' in parents" for p in c) + ")" for c in chunks]
        else:
            queries = [base]
        for q in queries:
            for f in self._paged_list(q, LIST_FIELDS):
                meta = f.get("imageMediaMetadata") or {}
                yield DrivePhoto(
                    id=f["id"],
                    name=f["name"],
                    mime_type=f["mimeType"],
                    md5=f.get("md5Checksum"),
                    thumbnail_link=f.get("thumbnailLink"),
                    created_time=f.get("createdTime"),
                    taken_time=meta.get("time"),
                )

    def fetch_image_bytes(self, photo: DrivePhoto, max_side: int = 1600) -> bytes:
        """썸네일(JPEG, 긴 변 max_side px)을 우선 받아 전송량을 줄이고, 없으면 원본을 받는다.

        썸네일은 HEIC/RAW도 JPEG로 변환해 주므로 디코딩 문제도 함께 해결된다.
        max_side <= 0 이면 항상 원본을 받는다.
        """
        if max_side > 0 and photo.thumbnail_link:
            url = photo.thumbnail_link.rsplit("=s", 1)[0] + f"=s{max_side}"
            try:
                r = self._retry(lambda: self.session.get(url, timeout=60))
                if r.status_code == 200 and r.content:
                    return r.content
            except Exception:
                pass
        return self.download_original(photo.id)

    def download_original(self, file_id: str) -> bytes:
        buf = io.BytesIO()
        req = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)
        downloader = MediaIoBaseDownload(buf, req, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _, done = self._retry(downloader.next_chunk)
        return buf.getvalue()

    def ensure_folder(self, name: str, parent_id: str | None = None) -> str:
        q = f"name = '{name}' and mimeType = '{FOLDER_MIME}' and trashed = false"
        if parent_id:
            q += f" and '{parent_id}' in parents"
        existing = next(iter(self._paged_list(q, "files(id)")), None)
        if existing:
            return existing["id"]
        body = {"name": name, "mimeType": FOLDER_MIME}
        if parent_id:
            body["parents"] = [parent_id]
        return self.service.files().create(body=body, fields="id").execute()["id"]

    def add_shortcut(self, target_id: str, name: str, folder_id: str) -> str:
        """원본을 옮기거나 복사하지 않고 결과 폴더에 바로가기만 만든다(용량 0)."""
        body = {
            "name": name,
            "mimeType": SHORTCUT_MIME,
            "shortcutDetails": {"targetId": target_id},
            "parents": [folder_id],
        }
        return self._retry(lambda: self.service.files().create(body=body, fields="id").execute())["id"]

    @staticmethod
    def _retry(fn, attempts: int = 5):
        for i in range(attempts):
            try:
                return fn()
            except HttpError as e:
                if e.resp.status not in (403, 429, 500, 502, 503, 504) or i == attempts - 1:
                    raise
            time.sleep(2 ** i)
