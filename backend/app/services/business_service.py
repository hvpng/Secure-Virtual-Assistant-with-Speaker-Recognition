"""Explicit M4 business operations; Gemini never invokes these directly."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from app.db.database import Employee


class BusinessServiceError(RuntimeError):
    pass


class EmployeeNotFoundError(BusinessServiceError):
    pass


def _employee(db: Session, employee_id: str) -> Employee:
    employee = db.get(Employee, employee_id)
    if employee is None:
        raise EmployeeNotFoundError(f"Không tìm thấy nhân viên {employee_id}.")
    return employee


def answer_faq(topic: str) -> str:
    normalized = " ".join(topic.lower().strip().split())
    if "vpn" in normalized:
        return "Để xin quyền VPN, hãy tạo yêu cầu trên cổng IT và chọn mục Truy cập từ xa."
    if "giờ" in normalized or "làm việc" in normalized:
        return "Giờ làm việc tiêu chuẩn là từ 8 giờ đến 17 giờ, thứ Hai đến thứ Sáu."
    if "nghỉ" in normalized or "lịch" in normalized:
        return "Lịch nghỉ công ty được công bố trên cổng thông tin nhân sự nội bộ."
    return "Vui lòng xem mục FAQ trên cổng IT hoặc liên hệ bộ phận hỗ trợ nội bộ."


def reset_password(db: Session, employee_id: str) -> str:
    employee = _employee(db, employee_id)
    employee.password_hash_mock = "mock-reset-required-on-next-login"
    return "Mật khẩu đã được đặt vào trạng thái cần đổi ở lần đăng nhập tiếp theo."


def check_salary_insurance(db: Session, employee_id: str) -> str:
    employee = _employee(db, employee_id)
    return (
        f"Lương hiện tại: {employee.salary_mock}. "
        f"Trạng thái bảo hiểm: {employee.insurance_status}."
    )


def check_leave_days(db: Session, employee_id: str) -> str:
    employee = _employee(db, employee_id)
    return f"Bạn còn {employee.leave_days_left} ngày phép."


def check_today_meetings(db: Session, employee_id: str) -> str:
    employee = _employee(db, employee_id)
    meetings = employee.meetings_today or []
    if not meetings:
        return "Hôm nay bạn không có lịch họp."
    return "Lịch họp hôm nay của bạn: " + "; ".join(meetings) + "."


BusinessCallable = Callable[[Session, str | None, dict[str, Any]], str]


def _faq_dispatch(db: Session, employee_id: str | None, arguments: dict[str, Any]) -> str:
    del db, employee_id
    return answer_faq(arguments["topic"])


def _reset_dispatch(db: Session, employee_id: str | None, arguments: dict[str, Any]) -> str:
    del arguments
    if employee_id is None:
        raise BusinessServiceError("Thiếu authenticated employee cho reset_password.")
    return reset_password(db, employee_id)


def _salary_dispatch(db: Session, employee_id: str | None, arguments: dict[str, Any]) -> str:
    del arguments
    if employee_id is None:
        raise BusinessServiceError("Thiếu authenticated employee cho salary/insurance.")
    return check_salary_insurance(db, employee_id)


def _leave_dispatch(db: Session, employee_id: str | None, arguments: dict[str, Any]) -> str:
    del arguments
    if employee_id is None:
        raise BusinessServiceError("Thiếu authenticated employee cho leave days.")
    return check_leave_days(db, employee_id)


def _meetings_dispatch(db: Session, employee_id: str | None, arguments: dict[str, Any]) -> str:
    del arguments
    if employee_id is None:
        raise BusinessServiceError("Thiếu authenticated employee cho meetings.")
    return check_today_meetings(db, employee_id)


BUSINESS_FUNCTIONS: dict[str, BusinessCallable] = {
    "answer_faq": _faq_dispatch,
    "reset_password": _reset_dispatch,
    "check_salary_insurance": _salary_dispatch,
    "check_leave_days": _leave_dispatch,
    "check_today_meetings": _meetings_dispatch,
}


def execute_business_function(
    db: Session,
    function_name: str,
    authenticated_employee_id: str | None,
    arguments: dict[str, Any],
) -> str:
    function = BUSINESS_FUNCTIONS.get(function_name)
    if function is None:
        raise BusinessServiceError("Business function không nằm trong allowlist.")
    return function(db, authenticated_employee_id, arguments)
