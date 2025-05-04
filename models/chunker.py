import re
import pandas as pd
from tqdm import tqdm
import logging
from config import Config
from typing import List, Dict, Tuple

class JSSplitter:
    def __init__(self):
        self.min_tokens = 50
        self.max_tokens = Config.CHUNK_SIZE
        self.code_pattern = re.compile(r'```(?:javascript)?\s*([\s\S]+?)\s*```')
        self.logger = logging.getLogger(__name__)
        
    def _count_tokens(self, text: str) -> int:
        """Đếm số từ trong text (ước lượng)"""
        return len(text.split())
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Trích xuất từ khóa JavaScript quan trọng"""
        js_keywords = {
            'function', 'var', 'let', 'const', 'return', 'if', 'else', 
            'for', 'while', 'class', 'import', 'export', 'this', 'async', 
            'await', 'try', 'catch', 'promise', 'array', 'object', 'string',
            'number', 'boolean', 'null', 'undefined', 'typeof', 'instanceof'
        }
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return list(set(word for word in words if word in js_keywords))[:5]
    
    def process_csv(self, file_path: str) -> List[Dict]:
        """Xử lý file CSV thành các chunks theo cấu trúc mới"""
        try:
            df = pd.read_csv(file_path)
            
            # Kiểm tra cột bắt buộc
            required_columns = {'bai', 'tieude', 'noidung', 'id'}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                raise ValueError(f"Thiếu các cột bắt buộc: {missing}")
            
            chunks = []
            
            for _, row in tqdm(df.iterrows(), desc=f"Processing {file_path}"):
                # Tạo nội dung kết hợp từ các trường
                header = f"Bài {row['bai']}: {row['tieude']}"
                content = f"{header}\n\n{row['noidung']}"
                
                # Xử lý và thêm metadata
                chunks.extend(self._split_content(content, {
                    'bai': row['bai'],
                    'tieude': row['tieude'],
                    'doc_id': row['id'],
                    'original_length': len(content)
                }))
            
            self.logger.info(f"Đã xử lý {len(chunks)} chunks từ {file_path}")
            return chunks
        except Exception as e:
            self.logger.error(f"Lỗi khi xử lý {file_path}: {str(e)}")
            return []
    
    def _split_content(self, text: str, metadata: Dict) -> List[Dict]:
        """Chia nội dung thành các chunks với metadata đầy đủ"""
        parts = []
        last_pos = 0
        
        # Tách code và text
        for match in self.code_pattern.finditer(text):
            if match.start() > last_pos:
                parts.append(('text', text[last_pos:match.start()].strip()))
            parts.append(('code', match.group(1).strip()))
            last_pos = match.end()
        
        # Phần còn lại
        remaining_text = text[last_pos:].strip()
        if remaining_text:
            parts.append(('text', remaining_text))
        
        # Gom nhóm thành chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for part_type, part_content in parts:
            if not part_content:
                continue
                
            part_tokens = self._count_tokens(part_content)
            
            # Nếu vượt quá max tokens và current_chunk không rỗng
            if (current_length + part_tokens > self.max_tokens and 
                current_length >= self.min_tokens and 
                current_chunk):
                chunks.append(self._create_chunk(current_chunk, metadata))
                current_chunk = []
                current_length = 0
            
            current_chunk.append((part_type, part_content))
            current_length += part_tokens
        
        # Thêm chunk cuối cùng nếu có
        if current_chunk and current_length >= self.min_tokens:
            chunks.append(self._create_chunk(current_chunk, metadata))
        
        return chunks
    
    def _create_chunk(self, parts: List[Tuple[str, str]], metadata: Dict) -> Dict:
        """Tạo dictionary chunk với metadata đầy đủ"""
        chunk_types = {part[0] for part in parts}
        chunk_type = 'mixed' if len(chunk_types) > 1 else chunk_types.pop()
        
        # Nối nội dung
        content = '\n\n'.join([p[1] for p in parts])
        
        return {
            'content': content,
            'type': chunk_type,
            'doc_id': metadata['doc_id'],
            'bai': metadata['bai'],
            'tieude': metadata['tieude'],
            'keywords': self._extract_keywords(content),
            'has_code': 'code' in chunk_types,
            'word_count': self._count_tokens(content),
            'original_length': metadata['original_length'],
            'is_truncated': self._count_tokens(content) < metadata['original_length']
        }