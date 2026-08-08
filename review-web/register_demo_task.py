"""把之前 FunASR 测试的成片注册为 web 任务，供在线预览"""
import sys, time
from pathlib import Path

sys.path.insert(0, r'C:\Users\MgAl\越南语自动化转译')
from services.pipeline import create_task

out = Path(r'C:\Users\MgAl\越南语自动化转译\output\最终成片\A_FunASR')
status = create_task(
    video_path=Path(r'C:\Users\MgAl\越南语自动化转译\testsuorse\第1集 (1).mp4'),
    episode_tag='FunASR_第1集',
    output_dir=out,
)
status.status = 'done'
status.stage = 'done'
status.progress = 100
status.message = 'FunASR + AI纠错 + 翻译 + AI校对 + 烧录 全流程完成'
status.finished_at = time.time()

final_video = str(out / '第1集_越南语转译成片.mp4')
clean_video = str(Path(r'C:\Users\MgAl\越南语自动化转译\output\消字幕测试\消字幕结果.mp4'))

status.artifacts = {
    'video': str(Path(r'C:\Users\MgAl\越南语自动化转译\testsuorse\第1集 (1).mp4')),
    'audio': str(out / '第1集_audio.wav'),
    'cn_srt': str(out / '第1集_中文字幕.srt'),
    'vi_srt_v1': str(out / '第1集_越南语初译.srt'),
    'vi_srt_final': str(out / '第1集_越南语终版.srt'),
    'review_docx': str(out / '第1集_中越双语审核.docx'),
    'transcript_raw': (out / '第1集_转写原始.txt').read_text(encoding='utf-8'),
    'transcript_corrected': (out / '第1集_转写纠错后.txt').read_text(encoding='utf-8'),
    'final_video': final_video,
    'clean_video': clean_video,
}
print('Task registered:', status.task_id)
print('Artifacts:', len(status.artifacts))
print('Final video:', final_video)
print('Exists:', Path(final_video).exists())
