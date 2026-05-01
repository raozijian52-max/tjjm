import os

from config import CONFIG
from stage1_read_and_manifest import scan_h5_files, is_h5_file, is_tar_file


# 测试当前 raw_dir 在阶段一读取逻辑下是否能读到文件
# 运行方式：python openlet_project/test_raw_dir_scan.py
def main():
    raw_dir = CONFIG["raw_dir"]
    abs_raw_dir = os.path.abspath(raw_dir)

    print("=" * 80)
    print("[RawDir Scan Test]")
    print(f"CONFIG['raw_dir'] = {raw_dir}")
    print(f"Absolute raw_dir  = {abs_raw_dir}")

    if not os.path.isdir(raw_dir):
        print("\n[FAIL] raw_dir 不存在或不是目录。")
        return

    all_entries = []
    for dir_path, _, file_names in os.walk(raw_dir):
        for file_name in file_names:
            all_entries.append(os.path.join(dir_path, file_name))

    h5_candidates = [p for p in all_entries if is_h5_file(os.path.basename(p))]
    tar_candidates = [p for p in all_entries if is_tar_file(os.path.basename(p))]

    print(f"\n目录下总文件数: {len(all_entries)}")
    print(f"直接 .h5 候选数: {len(h5_candidates)}")
    print(f"压缩包(.tar/.tgz/...)候选数: {len(tar_candidates)}")

    scanned = scan_h5_files()
    print(f"\n按 stage1 scan_h5_files() 实际可读取到的 H5 数量: {len(scanned)}")

    preview_n = 20
    if scanned:
        print(f"\n前 {min(preview_n, len(scanned))} 个样例路径:")
        for p in scanned[:preview_n]:
            print(f"  - {p}")
        if len(scanned) > preview_n:
            print(f"  ... 共 {len(scanned)} 个，以上仅展示前 {preview_n} 个")

        print("\n[PASS] 当前 raw_dir 可被阶段一读取逻辑识别到数据文件。")
    else:
        print("\n[FAIL] stage1 读取逻辑未识别到任何可用 H5。")
        print("建议检查：")
        print("1) raw_dir 路径是否写对（相对路径是否相对项目根）")
        print("2) 文件后缀是否为 .h5（大小写不敏感）")
        print("3) 若是压缩包，是否为 tar/tar.gz/tgz/tar.bz2/tbz2，且包内确实有 .h5")

    print("=" * 80)


if __name__ == "__main__":
    main()
