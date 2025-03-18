# from flask import Flask, render_template, request, redirect, url_for, session, send_file
# from flask_bootstrap import Bootstrap
# import torch
# from transformers import BertTokenizer, BertForMaskedLM
# import PyPDF2
# from PyPDF2 import PdfReader, PdfWriter
# import random
# from io import BytesIO
# from reportlab.pdfgen import canvas
# from builtins import enumerate

# app = Flask(__name__, template_folder="templates")

# app.secret_key = "secret"
# Bootstrap(app)

# # Load BERT model and tokenizer
# model_name = "bert-base-uncased"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForMaskedLM.from_pretrained(model_name)

# # def generate_mcqs(text, num_questions=5):
# #     sentences = text.split(". ")
# #     num_questions = min(num_questions, len(sentences))
# #     selected_sentences = random.sample(sentences, num_questions)
# #     mcqs = []

# #     for sentence in selected_sentences:
# #         words = sentence.split()
# #         if len(words) < 3:
# #             continue
        
# #         masked_index = random.randint(0, len(words) - 1)
# #         original_word = words[masked_index]
# #         words[masked_index] = "[MASK]"
# #         masked_sentence = " ".join(words)

# #         inputs = tokenizer(masked_sentence, return_tensors="pt")
# #         with torch.no_grad():
# #             outputs = model(**inputs)
        
# #         predictions = torch.topk(outputs.logits[0, masked_index], 10).indices.tolist()

# #         # Filtering out only meaningful words from BERT's suggestions
# #         answer_choices = set()
# #         for pred in predictions:
# #             predicted_word = tokenizer.decode([pred]).strip()
# #             if predicted_word.isalpha() and predicted_word.lower() != original_word.lower():
# #                 answer_choices.add(predicted_word)
# #             if len(answer_choices) >= 3:  # We need 3 wrong options
# #                 break

# #         # Ensure we always have 4 options (1 correct + 3 wrong)
# #         answer_choices = list(answer_choices)
# #         while len(answer_choices) < 3:
# #             answer_choices.append("Unknown")  # Fallback for missing wrong options

# #         answer_choices.append(original_word)
# #         random.shuffle(answer_choices)

# #         correct_answer = chr(65 + answer_choices.index(original_word))  # 'A', 'B', 'C', 'D'
        
# #         mcqs.append((masked_sentence.replace("[MASK]", "____"), answer_choices, correct_answer))
    
# #     return mcqs
# def generate_mcqs(text, num_questions=5):
#     sentences = text.split(". ")  # Simple sentence split
#     num_questions = min(num_questions, len(sentences))
#     selected_sentences = random.sample(sentences, num_questions)
#     mcqs = []

#     for sentence in selected_sentences:
#         words = sentence.split()
#         if len(words) < 3:
#             continue
#         masked_index = random.randint(0, len(words) - 1)
#         original_word = words[masked_index]
#         words[masked_index] = "[MASK]"
#         masked_sentence = " ".join(words)

#         inputs = tokenizer(masked_sentence, return_tensors="pt")
#         with torch.no_grad():
#             outputs = model(**inputs)
#         predictions = torch.topk(outputs.logits[0, masked_index], 4).indices
#         answer_choices = [original_word] + [tokenizer.decode([pred.item()]) for pred in predictions]
#         random.shuffle(answer_choices)
#         correct_answer = chr(65 + answer_choices.index(original_word))  # A, B, C, D

#         mcqs.append({
#             "question": masked_sentence.replace("[MASK]", "____"),
#             "options": answer_choices,
#             "answer": correct_answer
#         })

#     return mcqs




# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         text = ""

#         # Check if a file is uploaded
#         if "file" in request.files and request.files["file"].filename:
#             file = request.files["file"]
#             if file.filename.endswith(".pdf"):
#                 text = process_pdf(file)  # Extract text from PDF
#             else:
#                 text = file.read().decode("utf-8")  # Read text file content
#         else:
#             text = request.form.get("text", "").strip()  # Get text input from form
        
#         # Ensure text is not empty before proceeding
#         if not text:
#             return "Error: No text provided for MCQ generation", 400
        
#         num_questions = int(request.form.get("num_questions", 5))
#         mode = request.form.get("mode", "quiz")  # Default mode to quiz

#         # Generate MCQs and store in session
#         mcqs = generate_mcqs(text, num_questions)
#         if not mcqs:
#             return "Error: Unable to generate MCQs. Try a longer text.", 400

#         session["mcqs"] = mcqs
#         session["quiz_mode"] = mode  # Store mode in session
        
#         if mode == "only_mcqs":
#             return redirect(url_for("download_pdf"))
#         else:
#             return redirect(url_for("quiz"))

#     return render_template("index.html")


# # @app.route("/quiz", methods=["GET", "POST"])
# # def quiz():
# #     if "mcqs" not in session or not session["mcqs"]:  # Ensure MCQs exist
# #         return "Error: No MCQs available. Please generate them first.", 400
    
# #     mcqs = session["mcqs"]

# #     if request.method == "POST":
# #         user_answers = request.form.getlist("answer")
# #         correct_answers = [mcq[2] for mcq in mcqs]  # Extract correct answers
        
# #         score = sum(1 for i, ans in enumerate(user_answers) if ans == correct_answers[i])
# #         session["score"] = score  # Store score in session

# #         return redirect(url_for("result"))

# #     # Pass `enumerate` explicitly
# #     return render_template("quiz.html", mcqs=mcqs, enumerate=enumerate)
# # @app.route("/quiz", methods=["GET", "POST"])
# # def quiz():
# #     if "mcqs" not in session:
# #         return redirect(url_for("index"))

# #     if request.method == "POST":
# #         user_answers = []
# #         for i in range(len(session["mcqs"])):
# #             answer = request.form.get(f"answer_{i}")
# #             user_answers.append(answer)

# #         score = sum(1 for i, ans in enumerate(user_answers) if ans == session["mcqs"][i][2])
# #         session["score"] = score
# #         return redirect(url_for("result"))

# #     return render_template("quiz.html", mcqs=session["mcqs"])
# # @app.route("/quiz", methods=["GET", "POST"])
# # def quiz():
# #     if "mcqs" not in session or not session["mcqs"]:
# #         return redirect(url_for("index"))

# #     mcqs = session["mcqs"]

# #     if request.method == "POST":
# #         user_answers = []
# #         for i, (question, choices, correct_answer) in enumerate(mcqs):  # ✅ Unpack all three values
# #             answer = request.form.get(f"answer_{i}")  # Correctly fetch answers
# #             user_answers.append(answer if answer else "None")

# #         score = sum(1 for i, ans in enumerate(user_answers) if ans == mcqs[i][2])
# #         session["score"] = score
# #         return redirect(url_for("result"))

# #     return render_template("quiz.html", mcqs=mcqs, enumerate=enumerate)  # ✅ Pass enumerate
# # @app.route("/quiz", methods=["GET", "POST"])
# # def quiz():
# #     if "mcqs" not in session or not session["mcqs"]:
# #         return redirect(url_for("index"))

# #     mcqs = session["mcqs"]

# #     if request.method == "POST":
# #         user_answers = []
# #         for i, mcq in enumerate(mcqs):  # Correctly unpack the MCQ tuple
# #             question, choices, correct_answer = mcq  # Unpacking in the loop
# #             answer = request.form.get(f"answer_{i}")  # Fetch user's answer
# #             user_answers.append(answer if answer else "None")

# #         # Calculate the score
# #         score = sum(1 for i, ans in enumerate(user_answers) if ans == mcqs[i][2])
# #         session["score"] = score
# #         return redirect(url_for("result"))

# #     return render_template("quiz.html", mcqs=mcqs, enumerate=enumerate)

# @app.route("/quiz", methods=["GET", "POST"])
# def quiz():
#     if "mcqs" not in session:
#         return redirect(url_for("index"))

#     if request.method == "POST":
#         user_answers = request.form.getlist("answer")
#         mcqs = session["mcqs"]
#         score = 0

#         # Use a manual index instead of enumerate
#         for i in range(len(mcqs)):
#             correct_answer = mcqs[i]["answer"]
#             if i < len(user_answers) and user_answers[i] == correct_answer:
#                 score += 1

#         session["score"] = score
#         return redirect(url_for("result"))

#     return render_template("quiz.html", mcqs=session["mcqs"])








# @app.route("/result")
# def result():
#     return render_template("result.html", score=session.get("score", 0))




# @app.route("/download_pdf")
# def download_pdf():
#     mcqs = session.get("mcqs", [])
#     if not mcqs:
#         return "No MCQs available to generate PDF."

#     pdf_buffer = BytesIO()
#     pdf_canvas = canvas.Canvas(pdf_buffer)
#     pdf_canvas.setFont("Helvetica", 12)
    
#     y_position = 800  # Starting position from the top of the page

#     for i, (question, choices, _) in enumerate(mcqs, 1):
#         if y_position < 100:  # If there's no space left, create a new page
#             pdf_canvas.showPage()
#             pdf_canvas.setFont("Helvetica", 12)
#             y_position = 800

#         pdf_canvas.drawString(50, y_position, f"Q{i}: {question}")
#         y_position -= 20  # Move down for choices

#         for j, choice in enumerate(choices, 1):
#             pdf_canvas.drawString(70, y_position, f"{chr(64+j)}. {choice}")
#             y_position -= 20  # Space between options

#         y_position -= 20  # Extra space after each question

#     pdf_canvas.save()
#     pdf_buffer.seek(0)

#     return send_file(pdf_buffer, as_attachment=True, download_name="mcqs.pdf", mimetype="application/pdf")


# def process_pdf(file):
#     text = ""
#     pdf_reader = PdfReader(file)
#     for page in pdf_reader.pages:
#         text += page.extract_text() + "\n"
#     return text

# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, render_template, request, redirect, url_for, session, send_file
from flask_bootstrap import Bootstrap
import torch
from transformers import BertTokenizer, BertForMaskedLM
import random
from io import BytesIO
from reportlab.pdfgen import canvas
from PyPDF2 import PdfReader

app = Flask(__name__, template_folder="templates")

app.secret_key = "secret"
Bootstrap(app)

# Load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)


def generate_mcqs(text, num_questions=5):
    sentences = text.split(". ")  # Simple sentence split
    num_questions = min(num_questions, len(sentences))
    selected_sentences = random.sample(sentences, num_questions)
    mcqs = []

    for sentence in selected_sentences:
        words = sentence.split()
        if len(words) < 3:
            continue
        masked_index = random.randint(0, len(words) - 1)
        original_word = words[masked_index]
        words[masked_index] = "[MASK]"
        masked_sentence = " ".join(words)

        inputs = tokenizer(masked_sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.topk(outputs.logits[0, masked_index], 4).indices
        answer_choices = [original_word] + [tokenizer.decode([pred.item()]) for pred in predictions]
        random.shuffle(answer_choices)
        correct_answer = chr(65 + answer_choices.index(original_word))  # A, B, C, D

        mcqs.append({
            "question": masked_sentence.replace("[MASK]", "____"),
            "options": answer_choices,
            "answer": correct_answer
        })

    return mcqs


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = ""

        # Check if a file is uploaded
        if "file" in request.files and request.files["file"].filename:
            file = request.files["file"]
            if file.filename.endswith(".pdf"):
                text = process_pdf(file)  # Extract text from PDF
            else:
                text = file.read().decode("utf-8")  # Read text file content
        else:
            text = request.form.get("text", "").strip()  # Get text input from form

        # Ensure text is not empty before proceeding
        if not text:
            return "Error: No text provided for MCQ generation", 400

        num_questions = int(request.form.get("num_questions", 5))
        mode = request.form.get("mode", "quiz")  # Default mode to quiz

        # Generate MCQs and store in session
        mcqs = generate_mcqs(text, num_questions)
        if not mcqs:
            return "Error: Unable to generate MCQs. Try a longer text.", 400

        session["mcqs"] = mcqs
        session["quiz_mode"] = mode  # Store mode in session

        if mode == "only_mcqs":
            return redirect(url_for("download_pdf"))
        else:
            return redirect(url_for("quiz"))

    return render_template("index.html")



@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    if "mcqs" not in session or not session["mcqs"]:
        return redirect(url_for("index"))

    mcqs = session["mcqs"]

    if request.method == "POST":
        user_answers = []
        for i in range(len(mcqs)):
            user_answer = request.form.get(f"answer_{i}", "")  # Get answer for each question
            user_answers.append(user_answer)

        score = 0
        for i, mcq in enumerate(mcqs):
            correct_answer = mcq["answer"]
            if i < len(user_answers) and user_answers[i] == correct_answer:
                score += 1

        session["score"] = score
        return redirect(url_for("result"))

    return render_template("quiz.html", mcqs=mcqs, enumerate=enumerate)




@app.route("/result")
def result():
    return render_template("result.html", score=session.get("score", 0))


@app.route("/download_pdf")
def download_pdf():
    mcqs = session.get("mcqs", [])
    if not mcqs:
        return "No MCQs available to generate PDF."

    pdf_buffer = BytesIO()
    pdf_canvas = canvas.Canvas(pdf_buffer)
    pdf_canvas.setFont("Helvetica", 12)

    y_position = 800  # Starting position from the top of the page

    for i, mcq in enumerate(mcqs, 1):
        question = mcq["question"]
        choices = mcq["options"]

        if y_position < 100:  # If there's no space left, create a new page
            pdf_canvas.showPage()
            pdf_canvas.setFont("Helvetica", 12)
            y_position = 800

        pdf_canvas.drawString(50, y_position, f"Q{i}: {question}")
        y_position -= 20  # Move down for choices

        for j, choice in enumerate(choices, 1):
            pdf_canvas.drawString(70, y_position, f"{chr(64 + j)}. {choice}")
            y_position -= 20  # Space between options

        y_position -= 20  # Extra space after each question

    pdf_canvas.save()
    pdf_buffer.seek(0)

    return send_file(pdf_buffer, as_attachment=True, download_name="mcqs.pdf", mimetype="application/pdf")


def process_pdf(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text


if __name__ == "__main__":
    app.run(debug=True)
