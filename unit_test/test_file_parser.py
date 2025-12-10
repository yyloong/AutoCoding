from ms_agent.utils.file_parser_utils.file_parser import SingleFileParser
import os
import time
import logging

if __name__ == "__main__":
    
    # 1. 配置日志输出，确保能看到 "Start parsing" 和 "Hit cache"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    print("\n" + "="*60)
    print(" 🚀 开始真实数据测试 (Real Data Test) ")
    print("="*60)

    # 1. 设置待测试的真实文件路径 (支持 PDF, PPTX, DOCX 等)
    # Windows 用户请注意路径转义，例如 "D:\\Documents\\architecture_v1.pptx"
    TEST_FILE_PATH = "your test path"  # <-- 修改为你的本地文件路径

    # 2. 检查 API Key (如果你启用了 VLM 视觉分析)
    logger.warning("   如果你的文件包含图片且需要 VLM 解析流程图，请先设置 Key，否则 VLM 调用将失败。")
    
    if not os.path.exists(TEST_FILE_PATH):
        logger.error(f"❌ 测试文件不存在: {TEST_FILE_PATH}")
        logger.error("请修改代码中的 TEST_FILE_PATH 变量为真实的本地文件路径。")
        exit(1)

    workspace_dir = "your cache path"  # <-- 修改为你的缓存目录路径
    parser = SingleFileParser(cfg={
        'path': os.path.join(workspace_dir, 'parser_cache'),
        'structured_doc': True 
    })

    print(f"\n[Phase 1] 正在解析文件: {os.path.basename(TEST_FILE_PATH)}")
    print("⏳ 这可能需要几秒钟 (取决于文件大小和是否调用 VLM)...")
    
    t0 = time.time()
    input_params = {"url": TEST_FILE_PATH}
    
    try:
        # 调用 call 方法
        result_text = parser.call(input_params)
        t1 = time.time()
        
        print(f"✅ 解析完成! 耗时: {t1 - t0:.2f}s")
        print("-" * 30)
        print("📄 解析结果预览 (前 500 字符):")
        print(result_text)
        print("..." if len(result_text) > 500 else "")
        print("-" * 30)

    except Exception as e:
        logger.error(f"❌ 解析过程中发生错误: {e}")
        exit(1)

    # --- 测试阶段 2: 缓存命中测试 (读取本地 JSON) ---
    print(f"\n[Phase 2] 二次读取测试 (验证本地文件缓存机制)")
    
    t2 = time.time()
    cached_result = parser.call(input_params)
    t3 = time.time()

    print(f"⏱️ 二次读取耗时: {t3 - t2:.4f}s")
    
    if (t3 - t2) < 1.0:
        print("✅ 速度极快，成功命中本地 JSON 缓存。")
    else:
        print("⚠️ 速度较慢，可能未命中缓存，请检查日志。")

    # --- 验证内容一致性 ---
    if result_text == cached_result:
        print("✅ 内容一致性校验通过。")
    else:
        print("❌ 内容不一致!")

    # --- 检查 VLM 效果 (如果是 PPT/PDF) ---
    if "图片分析" in result_text or "Mermaid" in result_text or "Visual Analysis" in result_text:
        print("\n🎉 检测到 VLM 分析内容！流程图/架构图已被成功提取为文本描述。")
    else:
        print("\nℹ️ 未检测到明显的 VLM 分析标记。")
        print("   (原因可能是：文件中无图片、parse_pdf/ppt 中 extract_image 默认为 False、或 API 调用未生效)")

    print("\n测试结束。缓存文件位于:", parser.data_root)