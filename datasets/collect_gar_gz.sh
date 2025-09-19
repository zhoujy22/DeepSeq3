#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "用法: $0 压缩包路径 目标文件夹"
    exit 1
fi

TAR_FILE="$1"
DEST_DIR="$2"

if [ ! -f "$TAR_FILE" ]; then
    echo "错误：压缩包文件 '$TAR_FILE' 不存在"
    exit 2
fi

mkdir -p "$DEST_DIR"

tar -xf "$TAR_FILE" -C "$DEST_DIR"

echo "✅ 解压完成，文件已放入：$DEST_DIR"