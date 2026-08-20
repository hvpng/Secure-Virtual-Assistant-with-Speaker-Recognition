"""Idempotent demo employee data for local M4 usage."""

from sqlalchemy.orm import Session

from app.db.database import Employee


DEMO_EMPLOYEES: tuple[dict[str, object], ...] = (
    {
        "id": "NV001",
        "name": "Nguyễn Minh An",
        "leave_days_left": 12,
        "meetings_today": ["09:00 Họp nhóm IT", "14:00 Review bảo mật"],
        "salary_mock": "25.000.000 VND/tháng",
        "insurance_status": "Đang tham gia đầy đủ",
        "password_hash_mock": "mock-hash-nv001",
        "voice_enrolled": False,
    },
    {
        "id": "NV002",
        "name": "Trần Thu Bình",
        "leave_days_left": 7,
        "meetings_today": ["10:30 Họp dự án Phoenix"],
        "salary_mock": "22.000.000 VND/tháng",
        "insurance_status": "Đang tham gia đầy đủ",
        "password_hash_mock": "mock-hash-nv002",
        "voice_enrolled": False,
    },
    {
        "id": "NV003",
        "name": "Lê Hoàng Chi",
        "leave_days_left": 18,
        "meetings_today": [],
        "salary_mock": "28.500.000 VND/tháng",
        "insurance_status": "Đang chờ cập nhật hồ sơ",
        "password_hash_mock": "mock-hash-nv003",
        "voice_enrolled": False,
    },
    {
        "id": "NV004",
        "name": "Phạm Gia Dũng",
        "leave_days_left": 4,
        "meetings_today": ["08:30 Daily vận hành", "16:00 Họp khách hàng"],
        "salary_mock": "20.000.000 VND/tháng",
        "insurance_status": "Đang tham gia đầy đủ",
        "password_hash_mock": "mock-hash-nv004",
        "voice_enrolled": False,
    },
)


def seed_demo_employees(db: Session) -> None:
    changed = False
    for item in DEMO_EMPLOYEES:
        employee_id = str(item["id"])
        if db.get(Employee, employee_id) is None:
            db.add(Employee(**item))
            changed = True
    if changed:
        db.commit()
