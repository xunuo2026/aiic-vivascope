from pathlib import Path
import os
import sys

from fastapi.testclient import TestClient


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["DASHSCOPE_API_KEY"] = ""
os.environ["QWEN_MODEL"] = ""

from app.main import app


def main() -> None:
    client = TestClient(app)
    knowledge_payload = {
        "mode": "knowledge",
        "scenario": "保研复试",
        "style": "严格导师型",
        "major": "人工智能专业，大三，做过机器学习和计算机视觉课程项目",
        "target_profile": "保研申请智能感知方向导师，研究内容包括计算机视觉、模型泛化和可靠评估。",
        "project": "",
        "focus": "模型泛化、过拟合、评价指标",
    }
    response = client.post("/api/sessions", json=knowledge_payload)
    response.raise_for_status()
    knowledge_session = response.json()["session"]
    opening_question = knowledge_session["current_question"]
    banned_phrases = ["结合保研复试", "面试场景", "为什么可能被问到", "风险雷达", "知识点清单"]
    assert not any(phrase in opening_question for phrase in banned_phrases), opening_question
    response = client.post(f"/api/sessions/{knowledge_session['session_id']}/answer", json={"answer": "我不会"})
    response.raise_for_status()
    low_feedback = response.json()["turn"]["feedback"]
    low_score = low_feedback["score"]
    assert low_score <= 20, low_score
    assert low_feedback["strengths"] == [], low_feedback
    assert low_feedback["score_reason"], low_feedback
    assert low_feedback["rewrite"], low_feedback

    payload = {
        "mode": "mixed",
        "interview_length": "short",
        "scenario": "保研复试",
        "style": "严格导师型",
        "major": "人工智能专业，大三，做过机器学习和计算机视觉课程项目",
        "target_profile": "保研申请智能感知与机器人实验室，导师方向包括视觉检测、多模态感知和机器人操作。",
        "project": (
            "我参与了一个基于深度学习的实验室安全帽佩戴检测项目，目标是在实验室监控画面中识别人员是否正确佩戴安全帽。"
            "项目使用公开数据集和我们补充采集的少量实验室图片，先做数据清洗和标注，然后用 YOLO 系列目标检测模型训练。"
            "我的主要工作是整理数据、完成训练脚本、调整数据增强参数，并对比不同输入尺寸和置信度阈值下的检测效果。"
            "最终在测试集上 mAP 有一定提升，但在光照较暗和遮挡场景下仍然误检较多。"
        ),
        "focus": "训练方法选择、实验可靠性、个人贡献和创新点不足。",
    }
    response = client.post("/api/sessions", json=payload)
    response.raise_for_status()
    session = response.json()["session"]
    session_id = session["session_id"]
    assert session["max_rounds"] == 6

    standard_payload = payload | {"interview_length": "standard"}
    response = client.post("/api/sessions", json=standard_payload)
    response.raise_for_status()
    assert response.json()["session"]["max_rounds"] == 12

    answer = (
        "我的结论是这个项目主要解决实验室安全管理中的实时检测问题。"
        "我负责数据整理、训练脚本和参数对比，用 mAP、误检率和不同光照场景测试来验证，"
        "但数据量和遮挡场景仍是局限。"
    )
    for _ in range(session["max_rounds"]):
        response = client.post(f"/api/sessions/{session_id}/answer", json={"answer": answer})
        response.raise_for_status()
        session = response.json()["session"]

    assert session["status"] == "finished"
    assert session["final_report"]["total_score"] >= 0
    assert session["final_report"]["answer_summary"]
    assert session["final_report"]["overall_advice"]
    print("smoke-ok", session["final_report"]["total_score"])


if __name__ == "__main__":
    main()
