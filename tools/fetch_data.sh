#!/bin/sh
# Download the challenge data (Reaktor's Nice List) and link it into each
# solver directory. The data is not redistributed in this repository.
#   tools/fetch_data.sh
set -eu
cd "$(dirname "$0")/.."
URL=https://traveling-santa.reaktor.com/nicelist.txt
SHA256=a4ba73e54d2a7b4b750d2a475c34388894cc4ad153877850da2ffa7a1aa28c8e

mkdir -p data
if [ ! -f data/nicelist.txt ]; then
    curl -fsSL "$URL" -o data/nicelist.txt.part
    mv data/nicelist.txt.part data/nicelist.txt
fi
if command -v sha256sum >/dev/null 2>&1; then
    got=$(sha256sum data/nicelist.txt | cut -d' ' -f1)
else
    got=$(shasum -a 256 data/nicelist.txt | cut -d' ' -f1)
fi
if [ "$got" != "$SHA256" ]; then
    echo "data/nicelist.txt: checksum mismatch (got $got)" >&2
    echo "The results in this repository were computed on the file with sha256 $SHA256." >&2
    exit 1
fi

# Every solver reads nicelist.txt from its working directory.
for d in python rust c; do
    ln -sf ../data/nicelist.txt "$d/nicelist.txt"
done
echo "data/nicelist.txt OK (10,000 children), linked into python/ rust/ c/"
