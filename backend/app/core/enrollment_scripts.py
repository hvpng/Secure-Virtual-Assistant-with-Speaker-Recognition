"""Single source of truth for the seven server-controlled enrollment texts."""

ENROLLMENT_TEXTS: tuple[str, ...] = (
    "Tôi xác nhận đây là giọng nói của tôi để sử dụng trợ lý nội bộ.",
    "Hôm nay tôi đăng ký hồ sơ giọng nói cho hệ thống an toàn.",
    "Vui lòng hỗ trợ tôi truy cập dịch vụ công nghệ thông tin của công ty.",
    "Tôi thường sử dụng mạng riêng ảo khi làm việc từ xa.",
    "Thông tin lương và bảo hiểm cần được bảo vệ bằng xác thực giọng nói.",
    "Tôi muốn kiểm tra số ngày phép còn lại của mình.",
    "Trợ lý có thể cho tôi biết lịch họp trong ngày hôm nay.",
)


def enrollment_scripts_response() -> list[dict[str, object]]:
    return [
        {"index": index, "text": text}
        for index, text in enumerate(ENROLLMENT_TEXTS)
    ]
