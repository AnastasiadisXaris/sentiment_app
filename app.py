import gradio as gr
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

DATA_FILE = "training_data.csv"
vectorizer = None
model = None

# Αν δεν υπάρχει CSV, δημιουργούμε ένα κενό
if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=["text", "label"]).to_csv(DATA_FILE, index=False)

df = pd.read_csv(DATA_FILE)

def add_training_example(text, label):
    """Προσθήκη νέου δείγματος εκπαίδευσης και αποθήκευση."""
    global df
    if not text.strip():
        return "⚠️ Το κείμενο δεν μπορεί να είναι κενό."
    
    new_row = pd.DataFrame({"text": [text], "label": [label]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    return f"✅ Προστέθηκε δείγμα: '{text}' -> {label} (Σύνολο: {len(df)})"

def train_model():
    """Εκπαίδευση μοντέλου από αποθηκευμένα δεδομένα."""
    global model, vectorizer, df
    if len(df) < 3:
        return "⚠️ Χρειάζονται τουλάχιστον 3 δείγματα (ένα ανά κατηγορία)."
    
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(df["text"])
    model = MultinomialNB()
    model.fit(X_train, df["label"])
    return f"✅ Το μοντέλο εκπαιδεύτηκε με {len(df)} δείγματα."

def analyze_sentiment(file, manual_text):
    """Ανάλυση συναισθήματος από αρχείο ή κείμενο."""
    global model, vectorizer
    if model is None or vectorizer is None:
        return "⚠️ Δεν υπάρχει εκπαιδευμένο μοντέλο."
    
    if file is not None:
        text = file.read().decode("utf-8")
    elif manual_text.strip():
        text = manual_text
    else:
        return "⚠️ Δεν δόθηκαν δεδομένα."
    
    X_test = vectorizer.transform([text])
    prediction = model.predict(X_test)[0]
    return f"🔍 Ανάλυση: {prediction}"

with gr.Blocks() as demo:
    gr.Markdown("## 📝 Sentiment Analysis με Θετικό, Αρνητικό, Ουδέτερο")
    
    with gr.Tab("➕ Προσθήκη δεδομένων εκπαίδευσης"):
        text_input = gr.Textbox(label="Κείμενο", placeholder="Γράψε κείμενο...")
        label_input = gr.Dropdown(["Θετικό", "Αρνητικό", "Ουδέτερο"], label="Ετικέτα")
        add_button = gr.Button("Προσθήκη")
        output_add = gr.Textbox(label="Κατάσταση")
        add_button.click(fn=add_training_example, inputs=[text_input, label_input], outputs=output_add)
    
    with gr.Tab("⚙️ Εκπαίδευση μοντέλου"):
        train_button = gr.Button("Εκπαίδευση")
        output_train = gr.Textbox(label="Κατάσταση εκπαίδευσης")
        train_button.click(fn=train_model, inputs=[], outputs=output_train)
    
    with gr.Tab("🔍 Ανάλυση συναισθήματος"):
        file_input = gr.File(label="Φόρτωση αρχείου .txt")
        manual_text_input = gr.Textbox(label="Ή εισάγετε κείμενο χειροκίνητα")
        analyze_button = gr.Button("Ανάλυση")
        output_analysis = gr.Textbox(label="Αποτέλεσμα")
        analyze_button.click(fn=analyze_sentiment, inputs=[file_input, manual_text_input], outputs=output_analysis)

demo.launch()

