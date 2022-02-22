import re
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Dict, Union


@dataclass
class NewsgroupMessage:
    category: str
    headers: Dict[str, str]
    body: str

    @property
    def from_(self) -> str:
        return self.headers["From"]

    @property
    def subject(self) -> str:
        return self.headers["Subject"]

    @property
    def organization(self) -> str:
        return self.headers["Organization"]

    @property
    def lines(self) -> int:
        return int(self.headers["Lines"])


_HEADERS_PARSING_REGEX = re.compile(r"^(?P<header_name>\w+): (?P<header_value>.*)$", re.MULTILINE)


def read_newsgroup_message(filename: Union[str, PathLike[str]]) -> NewsgroupMessage:
    path = Path(filename)
    with path.open("r", encoding="latin-1") as f:
        raw_msg = f.read()
    headers_str, body = raw_msg.split("\n\n", maxsplit=1)

    if body[-1] != "\n":
        raise ValueError(f"Message body doesn't end with \\n in {filename}")
    body = body[:-1]  # strip ending newline

    return NewsgroupMessage(
        category=path.parent.name,
        headers=dict(_HEADERS_PARSING_REGEX.findall(headers_str)),
        body=body,
    )
