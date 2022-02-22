from pathlib import Path
from tempfile import TemporaryDirectory

from task1.newsgroup_message import read_newsgroup_message


def test_newsgroup_message_is_read_correctly():
    with TemporaryDirectory() as temp_dir:
        category_dir = Path(temp_dir) / "rec.autos"
        category_dir.mkdir()
        temp_file = category_dir / "123456"
        with temp_file.open("w", encoding="utf8") as f:
            f.write(_TEST_MESSAGE_BODY)

        result = read_newsgroup_message(temp_file)

        assert result.category == "rec.autos"
        assert result.lines == 22
        assert result.from_.startswith("maven")
        assert "NORTH" in result.organization
        assert "Miracle" in result.subject
        assert result.body.startswith("Re: Waving...")
        assert result.body.endswith("a wave.")


_TEST_MESSAGE_BODY = """From: maven@eskimo.com (Norman Hamer)
Subject: Re: A Miracle in California
Organization: -> ESKIMO NORTH (206) For-Ever <-
Lines: 22

Re: Waving...

I must say, that the courtesy of a nod or a wave.
"""
