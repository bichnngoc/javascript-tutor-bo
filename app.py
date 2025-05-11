import datetime
from flask import Flask, render_template, request, jsonify
import pandas as pd
from models.chunker import JSSplitter
from models.retriever import FAISSRetriever
from models.generator import GPTGenerator
from config import Config
import os
import logging
from flask import Flask, render_template, request, jsonify  # Thêm jsonify
from g4f.client import Client
# Đường dẫn file CSV
CHAT_HISTORY_CSV = "chat_history.csv"

app = Flask(__name__)
client = Client()
# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='chatbot.log'
)

app = Flask(__name__)

def initialize():
    """Khởi tạo dữ liệu với logging chi tiết"""
    try:
        if not os.path.exists(Config.DATA_DIR):
            logging.error(f"Thư mục {Config.DATA_DIR} không tồn tại")
            os.makedirs(Config.DATA_DIR)
            return None

        splitter = JSSplitter()
        retriever = FAISSRetriever()
        
        if not os.path.exists(Config.FAISS_INDEX_PATH):
            all_chunks = []
            for file in os.listdir(Config.DATA_DIR):
                if file.endswith('.csv'):
                    logging.info(f"Đang xử lý file: {file}")
                    file_path = os.path.join(Config.DATA_DIR, file)
                    chunks = splitter.process_csv(file_path)
                    all_chunks.extend(chunks)
                    logging.info(f"Đã thêm {len(chunks)} chunks từ {file}")
            
            if all_chunks:
                retriever.add_documents(all_chunks)
                logging.info(f"Đã tạo index với {len(all_chunks)} chunks")
        
        return retriever
    except Exception as e:
        logging.error(f"Lỗi khi khởi tạo: {str(e)}")
        return None

retriever = initialize()
if retriever is None:
    logging.warning("Không thể khởi tạo retriever, hệ thống sẽ hoạt động ở chế độ hạn chế")

generator = GPTGenerator()
def save_chat_history(question: str, answer: str):
    """
    Lưu lịch sử chat vào file CSV đơn giản
    Chỉ bao gồm: timestamp, question, answer
    """
    try:
        # Tạo dictionary với 3 trường cơ bản
        chat_record = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "answer": answer
        }
        
        # Tạo DataFrame từ record
        df = pd.DataFrame([chat_record])
        
        # Ghi vào file CSV (append nếu file đã tồn tại)
        df.to_csv(
            CHAT_HISTORY_CSV,
            mode='a',
            header=not os.path.exists(CHAT_HISTORY_CSV),
            index=False
        )
        
    except Exception as e:
        print(f"Lỗi khi lưu chat history: {str(e)}")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        question = data.get("question", "")
        language = data.get("lang", "vi")

        # Bước 1: Truy xuất thông tin liên quan từ FAISS
        context = retriever.search(question, top_k=3) if retriever else []
        
        # Bước 2: Tạo prompt với context
        if context:
            # Sử dụng generator với RAG
            answer = generator.generate(question, context, language)
        else:
            # Fallback nếu không có retriever
            answer = generator.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": question}],
            ).choices[0].message.content
            
        save_chat_history(question, answer)

        return jsonify({"response": answer})
    
    
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)