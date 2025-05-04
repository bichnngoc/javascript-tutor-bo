from g4f.client import Client
from config import Config
from typing import List, Dict
import logging

class GPTGenerator:
    def __init__(self):
        self.client = Client()
        self.model = "gpt-4"  # Sử dụng "gpt-4" thay vì "gpt-4-mini" để ổn định hơn
        self.retry_count = 3  # Số lần thử lại khi kết nối thất bại
        
        self.system_prompt = f"""
        Bạn là trợ lý JavaScript chuyên nghiệp. Hãy:
        1. Trả lời bằng {Config.DEFAULT_LANG}
        2. Giải thích đơn giản, dễ hiểu
        3. Luôn kèm ví dụ code khi có thể
        4. Dùng markdown để định dạng
        5. Nếu không chắc chắn, hãy nói "Theo tài liệu tham khảo..."
        """
    
    def generate(self, query: str, context: List[Dict], lang: str = None) -> str:
        lang = lang or Config.DEFAULT_LANG
        prompt = self._build_prompt(query, context, lang)
        
        for attempt in range(self.retry_count):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=Config.MAX_TOKENS,
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                logging.error(f"Lỗi lần {attempt + 1}: {str(e)}")
                if attempt == self.retry_count - 1:
                    return self._fallback_response(query, context, lang)
    
    def _build_prompt(self, query: str, context: List[Dict], lang: str) -> str:
        context_str = "\n\n".join(
            f"📌 Tài liệu {i+1} (Độ liên quan: {ctx['score']:.2f}):\n{ctx['content']}"
            for i, ctx in enumerate(context)
        )
        
        return f"""
        **Ngôn ngữ yêu cầu**: {lang}
        **Câu hỏi**: {query}
        
        **Thông tin tham khảo**:
        {context_str}
        
        **Hướng dẫn trả lời**:
        1. Ưu tiên thông tin có độ liên quan cao
        2. Giải thích như dạy người mới
        3. Kèm ví dụ code khi có thể
        4. Trả lời bằng {lang}
        """
    
    def _fallback_response(self, query: str, context: List[Dict], lang: str) -> str:
        """Phản hồi dự phòng khi không kết nối được GPT"""
        return f"""
        ⚠️ Hiện không kết nối được với GPT-4. Dưới đây là thông tin từ tài liệu:
        
        **Câu hỏi**: {query}
        
        **Tài liệu tham khảo**:
        {', '.join([ctx['doc_id'] for ctx in context])}
        
        Vui lòng thử lại sau hoặc kiểm tra kết nối Internet.
        """