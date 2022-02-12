import click
import pathlib
from pathlib import Path

from projects.abazarova.source.classifier.tokenizer import tokenize
from projects.abazarova.source.classifier.reader import read_emoji_to_label_mapping


@click.group()
def main():
    pass


@main.command()
@click.argument("text", type=str)
def label(text):
    path = Path(str(Path(Path.cwd()))[:-len("source.classifier")],"assets","emoji-to-label.yml")
    print(path)
    labels = read_emoji_to_label_mapping(path).classify(text)

    if len(labels) == 0:
        print("No labels found for the given text")
    else:
        print(f"Provided text mentions {', '.join(labels)}")

if __name__ == "__main__":
    main()
