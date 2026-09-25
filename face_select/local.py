"""로컬 폴더 모드: 드라이브에서 폴더째 내려받은 사진을 같은 파이프라인으로 처리한다.

API로 목록을 읽을 수 없는 폴더(링크로만 공유된 폴더 등)의 우회 경로.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .faces import IMAGE_EXTS

LOCAL_PREFIX = "local:"


@dataclass
class LocalPhoto:
    id: str
    name: str
    md5: str  # 실제 해시 대신 크기+수정시각으로 변경 여부만 판단
    taken_time: str | None = None

    @property
    def path(self) -> Path:
        return Path(self.id[len(LOCAL_PREFIX):])


def is_local(photo_id: str) -> bool:
    return photo_id.startswith(LOCAL_PREFIX)


def local_path(photo_id: str) -> Path:
    return Path(photo_id[len(LOCAL_PREFIX):])


def list_local_photos(root: str | Path) -> Iterator[LocalPhoto]:
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            path = Path(dirpath, fn).resolve()
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            st = path.stat()
            yield LocalPhoto(id=LOCAL_PREFIX + str(path), name=fn, md5=f"{st.st_size}-{st.st_mtime_ns}")
