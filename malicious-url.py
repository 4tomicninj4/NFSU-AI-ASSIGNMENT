import os
import sys
import time
import threading
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse

# Machine Learning imports
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report
    import joblib
except ImportError:
    print("ERROR: Required ML libraries not found!")
    print("Please install: pip install pandas numpy scikit-learn joblib")
    sys.exit(1)

# GUI imports
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
from tkinter.font import Font


class URLFeatureExtractor:
    """Extract features from URLs for ML classification"""

    @staticmethod
    def extract_features(url: str):
        """Extract numerical and statistical features from URL"""
        features = {}

        try:
            # Basic URL properties
            features["url_length"] = len(url)
            features["num_digits"] = sum(c.isdigit() for c in url)
            features["num_letters"] = sum(c.isalpha() for c in url)
            features["num_special_chars"] = sum(not c.isalnum() for c in url)
            features["num_dots"] = url.count(".")
            features["num_hyphens"] = url.count("-")
            features["num_underscores"] = url.count("_")
            features["num_slashes"] = url.count("/")
            features["num_question_marks"] = url.count("?")
            features["num_equal"] = url.count("=")
            features["num_at"] = url.count("@")
            features["num_ampersand"] = url.count("&")
            features["num_exclamation"] = url.count("!")
            features["num_space"] = url.count(" ")
            features["num_tilde"] = url.count("~")
            features["num_comma"] = url.count(",")
            features["num_plus"] = url.count("+")
            features["num_asterisk"] = url.count("*")
            features["num_hashtag"] = url.count("#")
            features["num_dollar"] = url.count("$")
            features["num_percent"] = url.count("%")

            # Parse URL
            parsed = urlparse(url)

            # Domain features
            domain = parsed.netloc
            features["domain_length"] = len(domain)
            features["has_ip"] = int(bool(re.match(r"\d+\.\d+\.\d+\.\d+", domain)))
            features["num_subdomains"] = domain.count(".") - 1 if domain else 0

            # Path features
            path = parsed.path
            features["path_length"] = len(path)
            features["num_path_tokens"] = len(path.split("/")) - 1

            # Query features
            query = parsed.query
            features["query_length"] = len(query)
            features["num_query_params"] = query.count("&") + 1 if query else 0

            # Protocol features
            features["is_https"] = int(parsed.scheme == "https")
            features["is_http"] = int(parsed.scheme == "http")

            # Suspicious patterns
            features["has_double_slash"] = int("//" in path)
            features["has_suspicious_words"] = int(
                any(
                    word in url.lower()
                    for word in [
                        "login",
                        "signin",
                        "bank",
                        "account",
                        "update",
                        "verify",
                        "secure",
                        "ebay",
                        "paypal",
                        "amazon",
                        "microsoft",
                        "apple",
                        "admin",
                    ]
                )
            )

            # Entropy (randomness measure)
            features["url_entropy"] = URLFeatureExtractor.calculate_entropy(url)
            features["domain_entropy"] = URLFeatureExtractor.calculate_entropy(domain)

            # Length ratios
            features["digits_ratio"] = features["num_digits"] / max(
                features["url_length"], 1
            )
            features["letters_ratio"] = features["num_letters"] / max(
                features["url_length"], 1
            )
            features["special_chars_ratio"] = features["num_special_chars"] / max(
                features["url_length"], 1
            )

            # Word features
            words = re.split(r"[^a-zA-Z0-9]", url)
            word_lengths = [len(w) for w in words if w]
            features["longest_word_length"] = max(word_lengths) if word_lengths else 0
            features["average_word_length"] = (
                float(np.mean(word_lengths)) if word_lengths else 0.0
            )

            # TLD (Top Level Domain)
            tld = domain.split(".")[-1] if domain and "." in domain else ""
            common_tlds = [
                "com",
                "org",
                "net",
                "edu",
                "gov",
                "co",
                "uk",
                "de",
                "fr",
                "in",
            ]
            features["has_common_tld"] = int(tld in common_tlds)
            features["tld_length"] = len(tld)

            # URL shortener detection
            shorteners = ["bit.ly", "goo.gl", "tinyurl", "t.co", "ow.ly", "buff.ly"]
            features["is_shortened"] = int(
                any(short in domain.lower() for short in shorteners)
            )

        except Exception as e:
            print(f"Error extracting features: {e}")
            # In case of error, return dummy numeric features
            return {f"feature_{i}": 0 for i in range(40)}

        return features

    @staticmethod
    def calculate_entropy(text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0
        probs = [float(text.count(c)) / len(text) for c in dict.fromkeys(text)]
        return float(-sum(p * np.log2(p) for p in probs))


class MaliciousURLDetector:
    """ML model for malicious URL detection"""

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_trained = False

    def train_model(self, csv_path, progress_callback=None):
        """Train the malicious URL detection model on Kaggle dataset"""
        try:
            if progress_callback:
                progress_callback(5, "Loading dataset...")
            df = pd.read_csv(csv_path)

            # Expect Kaggle columns: 'url', 'type'
            url_col = "url" if "url" in df.columns else df.columns[0]
            label_col = "type" if "type" in df.columns else df.columns[1]

            X_urls = df[url_col].astype(str).values
            y = df[label_col].astype(str).values

            # Encode labels
            self.label_encoder.fit(y)
            y_enc = self.label_encoder.transform(y)

            if progress_callback:
                progress_callback(15, "Extracting numeric features...")

            features_list = []
            total = len(X_urls)
            for i, u in enumerate(X_urls):
                if progress_callback and i % 2000 == 0:
                    progress_callback(15 + (i / max(total, 1)) * 20, f"Extracting {i}/{total} URLs...")
                features_list.append(URLFeatureExtractor.extract_features(u))

            X_num = pd.DataFrame(features_list)
            self.feature_names = list(X_num.columns)

            if progress_callback:
                progress_callback(40, "Building TF-IDF features...")

            # Char-level TF-IDF
            self.vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 4),
                max_features=300,
                min_df=2,
            )
            X_tfidf = self.vectorizer.fit_transform(X_urls).toarray()

            # Combine numeric + TF-IDF
            X_combined = np.hstack([X_num.values, X_tfidf])

            if progress_callback:
                progress_callback(55, "Splitting train/test...")

            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y_enc, test_size=0.2, random_state=42, stratify=y_enc
            )

            if progress_callback:
                progress_callback(65, "Scaling features...")

            X_train_sc = self.scaler.fit_transform(X_train)
            X_test_sc = self.scaler.transform(X_test)

            if progress_callback:
                progress_callback(75, "Training RandomForest model...")

            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )
            self.model.fit(X_train_sc, y_train)

            if progress_callback:
                progress_callback(90, "Evaluating model...")

            y_pred = self.model.predict(X_test_sc)
            acc = accuracy_score(y_test, y_pred)

            print("\n" + "=" * 60)
            print("Model Training Complete")
            print("=" * 60)
            print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
            print("\nClassification Report:")
            print(
                classification_report(
                    y_test, y_pred, target_names=self.label_encoder.classes_
                )
            )

            self.is_trained = True

            if progress_callback:
                progress_callback(100, "Training complete!")

            return acc

        except Exception as e:
            print(f"[TRAIN ERROR] {e}")
            raise

    def save_model(self, path="malicious_url_model.pkl"):
        data = {
            "model": self.model,
            "vectorizer": self.vectorizer,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
        }
        joblib.dump(data, path)
        print(f"Model saved to {path}")

    def load_model(self, path="malicious_url_model.pkl"):
        if not os.path.exists(path):
            return False
        data = joblib.load(path)
        self.model = data["model"]
        self.vectorizer = data["vectorizer"]
        self.scaler = data["scaler"]
        self.label_encoder = data["label_encoder"]
        self.feature_names = data["feature_names"]
        self.is_trained = True
        print(f"Model loaded from {path}")
        return True

    def _build_feature_vector(self, url: str):
        """Build combined [numeric + tfidf] feature vector for one URL"""
        f_dict = URLFeatureExtractor.extract_features(url)
        numeric = np.array([f_dict[name] for name in self.feature_names]).reshape(1, -1)
        tfidf = self.vectorizer.transform([url]).toarray()
        combined = np.hstack([numeric, tfidf])
        return self.scaler.transform(combined)

    def predict(self, url: str):
        if not self.is_trained:
            raise RuntimeError("Model not trained or loaded")
        X = self._build_feature_vector(url)
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        label = self.label_encoder.inverse_transform([pred])[0]
        return label, proba, pred

    def batch_predict(self, urls, progress_callback=None):
        results = []
        total = len(urls)
        for i, url in enumerate(urls):
            if progress_callback and i % 10 == 0:
                progress = (i / max(total, 1)) * 100
                progress_callback(progress, f"Checking {i}/{total} URLs...")
            try:
                label, proba, pred_idx = self.predict(url)
                is_mal = label.lower() in [
                    "malicious",
                    "phishing",
                    "defacement",
                    "malware",
                    "bad",
                ]
                conf = float(proba[pred_idx])
                risk = "HIGH" if conf > 0.8 else "MEDIUM" if conf > 0.5 else "LOW"
                results.append(
                    {
                        "url": url,
                        "prediction": label,
                        "is_malicious": is_mal,
                        "confidence": conf,
                        "risk_level": risk,
                        "probability": proba,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "url": url,
                        "prediction": "ERROR",
                        "is_malicious": False,
                        "confidence": 0.0,
                        "risk_level": "UNKNOWN",
                        "error": str(e),
                    }
                )
        if progress_callback:
            progress_callback(100, "Batch prediction complete!")
        return results


class URLScannerGUI:
    """GUI Application for Malicious URL Scanner"""

    def __init__(self, root):
        self.root = root
        self.root.title("Malicious URL Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1a1a2e")

        self.detector = MaliciousURLDetector()
        self.scan_results = []

        self._build_ui()
        self.load_default_model()

    # ======================== UI BUILD ========================

    def _build_ui(self):
        title_font = Font(family="Helvetica", size=22, weight="bold")
        tk.Label(
            self.root,
            text="🛡️ Malicious URL Detector",
            font=title_font,
            bg="#1a1a2e",
            fg="#00d9ff",
        ).pack(pady=20)

        main = tk.Frame(self.root, bg="#1a1a2e")
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # LEFT PANEL
        left = tk.Frame(main, bg="#16213e", relief=tk.RAISED, borderwidth=2)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10), pady=0)

        # URL SCANNER
        input_frame = tk.LabelFrame(
            left,
            text="🔍 URL Scanner",
            bg="#16213e",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=10,
            pady=10,
        )
        input_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        tk.Label(
            input_frame,
            text="Enter URL to check:",
            bg="#16213e",
            fg="#cccccc",
            font=("Arial", 10),
        ).pack(anchor="w")

        row = tk.Frame(input_frame, bg="#16213e")
        row.pack(fill=tk.X, pady=5)

        self.url_entry = tk.Entry(
            row, font=("Arial", 11), bg="#0f3460", fg="white", insertbackground="white"
        )
        self.url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        self.url_entry.bind("<Return>", lambda e: self.check_single_url())

        tk.Button(
            row,
            text="Check",
            command=self.check_single_url,
            bg="#00d9ff",
            fg="black",
            font=("Arial", 10, "bold"),
            relief=tk.FLAT,
            padx=10,
        ).pack(side=tk.RIGHT, padx=(5, 0))

        # MODEL MANAGEMENT – PUT HIGH UP SO YOU SEE IT
        model_frame = tk.LabelFrame(
            left,
            text="🤖 Model Management",
            bg="#16213e",
            fg="white",
            font=("Arial", 11, "bold"),
            padx=10,
            pady=10,
        )
        model_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(
            model_frame,
            text="📁 Load Model",
            command=self.load_model,
            bg="#5a5a5a",
            fg="white",
            relief=tk.FLAT,
            padx=15,
            pady=5,
        ).pack(fill=tk.X, pady=2)

        tk.Button(
            model_frame,
            text="📊 Train New Model",
            command=self.train_model_dialog,
            bg="#5a5a5a",
            fg="white",
            relief=tk.FLAT,
            padx=15,
            pady=5,
        ).pack(fill=tk.X, pady=2)

        self.model_status_label = tk.Label(
            model_frame,
            text="Model: Not Loaded",
            bg="#16213e",
            fg="#ff4444",
            font=("Arial", 9),
        )
        self.model_status_label.pack(pady=(5, 0))

        # BATCH SCAN
        batch_frame = tk.LabelFrame(
            left,
            text="📋 Batch Scanning",
            bg="#16213e",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=10,
            pady=10,
        )
        batch_frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)

        tk.Label(
            batch_frame,
            text="Enter multiple URLs (one per line):",
            bg="#16213e",
            fg="#cccccc",
            font=("Arial", 10),
        ).pack(anchor="w")

        self.batch_text = scrolledtext.ScrolledText(
            batch_frame,
            height=8,
            bg="#0f3460",
            fg="white",
            font=("Courier", 9),
            insertbackground="white",
        )
        self.batch_text.pack(fill=tk.BOTH, expand=True, pady=5)

        row2 = tk.Frame(batch_frame, bg="#16213e")
        row2.pack(fill=tk.X, pady=5)

        self.batch_btn = tk.Button(
            row2,
            text="🔍 Scan All URLs",
            command=self.batch_scan,
            bg="#00d9ff",
            fg="black",
            font=("Arial", 10, "bold"),
            relief=tk.FLAT,
            padx=15,
            pady=5,
        )
        self.batch_btn.pack(side=tk.LEFT, padx=(0, 5))

        tk.Button(
            row2,
            text="📂 Load from File",
            command=self.load_urls_from_file,
            bg="#5a5a5a",
            fg="white",
            relief=tk.FLAT,
            padx=15,
            pady=5,
        ).pack(side=tk.LEFT)

        # PROGRESS
        prog_frame = tk.Frame(left, bg="#16213e")
        prog_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(
            prog_frame,
            text="Progress:",
            bg="#16213e",
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w")

        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(
            prog_frame, variable=self.progress_var, maximum=100
        ).pack(fill=tk.X, pady=3)
        self.progress_label = tk.Label(
            prog_frame,
            text="Ready",
            bg="#16213e",
            fg="#00d9ff",
            font=("Arial", 9),
        )
        self.progress_label.pack(anchor="w")

        # STATS
        stats_frame = tk.LabelFrame(
            left,
            text="📊 Statistics",
            bg="#16213e",
            fg="white",
            font=("Arial", 11, "bold"),
        )
        stats_frame.pack(fill=tk.X, padx=10, pady=5)

        self.stats_labels = {}
        for name, color in [
            ("Total Scanned", "#00d9ff"),
            ("Malicious", "#ff4444"),
            ("Safe", "#44ff44"),
            ("Unknown", "#ffaa44"),
        ]:
            f = tk.Frame(stats_frame, bg="#16213e")
            f.pack(fill=tk.X, padx=10, pady=2)
            tk.Label(
                f,
                text=f"{name}:",
                bg="#16213e",
                fg="#cccccc",
                font=("Arial", 9),
            ).pack(side=tk.LEFT)
            lbl = tk.Label(f, text="0", bg="#16213e", fg=color, font=("Arial", 10, "bold"))
            lbl.pack(side=tk.RIGHT)
            self.stats_labels[name] = lbl

        # RIGHT PANEL
        right = tk.Frame(main, bg="#16213e", relief=tk.RAISED, borderwidth=2)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(
            right,
            text="🔍 Scan Results",
            bg="#16213e",
            fg="white",
            font=("Arial", 14, "bold"),
        ).pack(pady=10)

        tree_frame = tk.Frame(right, bg="#16213e")
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")

        self.results_tree = ttk.Treeview(
            tree_frame,
            columns=("URL", "Status", "Type", "Confidence", "Risk"),
            show="tree headings",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
        )
        vsb.config(command=self.results_tree.yview)
        hsb.config(command=self.results_tree.xview)

        self.results_tree.heading("#0", text="#")
        self.results_tree.heading("URL", text="URL")
        self.results_tree.heading("Status", text="Status")
        self.results_tree.heading("Type", text="Type")
        self.results_tree.heading("Confidence", text="Confidence")
        self.results_tree.heading("Risk", text="Risk Level")

        self.results_tree.column("#0", width=40)
        self.results_tree.column("URL", width=320)
        self.results_tree.column("Status", width=110)
        self.results_tree.column("Type", width=100)
        self.results_tree.column("Confidence", width=100)
        self.results_tree.column("Risk", width=100)

        self.results_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        self.results_tree.tag_configure(
            "malicious", background="#ff4444", foreground="white"
        )
        self.results_tree.tag_configure(
            "safe", background="#44ff44", foreground="black"
        )
        self.results_tree.tag_configure(
            "unknown", background="#ffaa44", foreground="black"
        )

        # ACTIONS + LOG
        btn_row = tk.Frame(right, bg="#16213e")
        btn_row.pack(fill=tk.X, padx=10, pady=(0, 5))
        tk.Button(
            btn_row,
            text="🗑️ Clear Results",
            command=self.clear_results,
            bg="#d9534f",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=5,
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            btn_row,
            text="💾 Export Results",
            command=self.export_results,
            bg="#5a5a5a",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=5,
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            btn_row,
            text="📊 View Statistics",
            command=self.show_statistics,
            bg="#5a5a5a",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=5,
        ).pack(side=tk.LEFT, padx=5)

        log_frame = tk.LabelFrame(
            right,
            text="📝 Activity Log",
            bg="#16213e",
            fg="white",
            font=("Arial", 10, "bold"),
        )
        log_frame.pack(fill=tk.BOTH, padx=10, pady=(0, 10))

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=6, bg="#0f0f0f", fg="#00ff00", font=("Courier", 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # ======================== LOG & MODEL ========================

    def log(self, msg, level="INFO"):
        t = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{t}] [{level}] {msg}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def load_default_model(self):
        path = "malicious_url_model.pkl"
        if os.path.exists(path) and self.detector.load_model(path):
            self.model_status_label.config(text="Model: Loaded ✓", fg="#44ff44")
            self.log("Loaded default model", "SUCCESS")
        else:
            self.log("No default model. Please train or load one.", "WARNING")

    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
        )
        if not file_path:
            return
        if self.detector.load_model(file_path):
            self.model_status_label.config(text="Model: Loaded ✓", fg="#44ff44")
            self.log("Model loaded successfully", "SUCCESS")
            messagebox.showinfo("Success", "Model loaded successfully.")
        else:
            self.log("Failed to load model", "ERROR")
            messagebox.showerror("Error", "Failed to load model.")

    def train_model_dialog(self):
        csv_path = filedialog.askopenfilename(
            title="Select Training Dataset (CSV)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not csv_path:
            return

        self.log("Starting training...", "INFO")

        def progress_cb(p, text):
            self.root.after(0, lambda: self.progress_var.set(p))
            self.root.after(0, lambda: self.progress_label.config(text=text))

        def worker():
            try:
                acc = self.detector.train_model(csv_path, progress_cb)
                save_path = filedialog.asksaveasfilename(
                    title="Save Trained Model",
                    defaultextension=".pkl",
                    filetypes=[("Pickle files", "*.pkl")],
                )
                if save_path:
                    self.detector.save_model(save_path)
                    self.root.after(
                        0,
                        lambda: self.model_status_label.config(
                            text="Model: Trained ✓", fg="#44ff44"
                        ),
                    )
                    self.root.after(
                        0,
                        lambda: messagebox.showinfo(
                            "Success",
                            f"Model trained successfully!\nAccuracy: {acc:.2%}\nSaved to: {save_path}",
                        ),
                    )
                self.root.after(0, lambda: self.progress_var.set(0))
                self.root.after(0, lambda: self.progress_label.config(text="Ready"))
            except Exception as e:
                self.root.after(0, lambda: self.log(f"Training failed: {e}", "ERROR"))
                self.root.after(
                    0, lambda: messagebox.showerror("Error", f"Training failed:\n{e}")
                )
                self.root.after(0, lambda: self.progress_var.set(0))
                self.root.after(
                    0, lambda: self.progress_label.config(text="Training failed")
                )

        threading.Thread(target=worker, daemon=True).start()

    # ======================== SCAN FUNCTIONS ========================

    def check_single_url(self):
        if not self.detector.is_trained:
            messagebox.showerror("Model Not Loaded", "Please load or train a model first.")
            return
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showwarning("No URL", "Please enter a URL.")
            return

        self.log(f"Checking URL: {url}", "INFO")
        try:
            label, proba, pred_idx = self.detector.predict(url)
            is_mal = label.lower() in [
                "malicious",
                "phishing",
                "defacement",
                "malware",
                "bad",
            ]
            conf = float(proba[pred_idx])
            risk = "HIGH" if conf > 0.8 else "MEDIUM" if conf > 0.5 else "LOW"
            status = "🔴 MALICIOUS" if is_mal else "🟢 SAFE"
            tag = "malicious" if is_mal else "safe"

            idx = len(self.scan_results) + 1
            display_url = url if len(url) <= 50 else url[:50] + "..."
            self.results_tree.insert(
                "",
                0,
                text=str(idx),
                values=(display_url, status, label, f"{conf:.2%}", risk),
                tags=(tag,),
            )

            self.scan_results.append(
                {
                    "url": url,
                    "prediction": label,
                    "is_malicious": is_mal,
                    "confidence": conf,
                    "risk_level": risk,
                    "probability": proba,
                }
            )
            self.update_statistics()

            if is_mal:
                self.log(
                    f"MALICIOUS URL: {label} ({conf:.2%})",
                    "WARNING",
                )
                messagebox.showwarning(
                    "Malicious URL Detected",
                    f"This URL appears MALICIOUS.\n\nType: {label}\nConfidence: {conf:.2%}\nRisk: {risk}",
                )
            else:
                self.log(
                    f"URL appears safe: {label} ({conf:.2%})",
                    "SUCCESS",
                )
                messagebox.showinfo(
                    "Result",
                    f"This URL appears SAFE.\n\nType: {label}\nConfidence: {conf:.2%}",
                )

            self.url_entry.delete(0, tk.END)
        except Exception as e:
            self.log(f"Error checking URL: {e}", "ERROR")
            messagebox.showerror("Error", f"Error checking URL:\n{e}")

    def batch_scan(self):
        if not self.detector.is_trained:
            messagebox.showerror("Model Not Loaded", "Please load or train a model first.")
            return
        text = self.batch_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("No URLs", "Enter URLs (one per line).")
            return
        urls = [u.strip() for u in text.splitlines() if u.strip()]

        self.log(f"Starting batch scan of {len(urls)} URLs...", "INFO")
        self.batch_btn.config(state=tk.DISABLED, text="⏳ Scanning...")

        def progress_cb(p, msg):
            self.root.after(0, lambda: self.progress_var.set(p))
            self.root.after(0, lambda: self.progress_label.config(text=msg))

        def worker():
            try:
                results = self.detector.batch_predict(urls, progress_cb)
                for r in results:
                    idx = len(self.scan_results) + 1
                    if "error" in r:
                        status = "❌ ERROR"
                        tag = "unknown"
                        label = "ERROR"
                        conf = 0.0
                        risk = "UNKNOWN"
                    else:
                        status = "🔴 MALICIOUS" if r["is_malicious"] else "🟢 SAFE"
                        tag = "malicious" if r["is_malicious"] else "safe"
                        label = r["prediction"]
                        conf = float(r["confidence"])
                        risk = r["risk_level"]
                    display_url = (
                        r["url"] if len(r["url"]) <= 50 else r["url"][:50] + "..."
                    )
                    self.root.after(
                        0,
                        lambda n=idx, u=display_url, s=status, l=label, c=conf, rk=risk, tg=tag: self.results_tree.insert(
                            "",
                            0,
                            text=str(n),
                            values=(u, s, l, f"{c:.2%}", rk),
                            tags=(tg,),
                        ),
                    )
                    self.scan_results.append(r)
                self.root.after(0, self.update_statistics)
                mal_count = sum(1 for r in results if r.get("is_malicious", False))
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Batch Scan Complete",
                        f"Scanned: {len(urls)}\nMalicious: {mal_count}\nSafe: {len(urls) - mal_count}",
                    ),
                )
                self.root.after(0, lambda: self.progress_var.set(0))
                self.root.after(0, lambda: self.progress_label.config(text="Ready"))
            except Exception as e:
                self.root.after(0, lambda: self.log(f"Batch scan failed: {e}", "ERROR"))
                self.root.after(
                    0,
                    lambda: messagebox.showerror("Error", f"Batch scan failed:\n{e}"),
                )
            finally:
                self.root.after(
                    0,
                    lambda: self.batch_btn.config(
                        state=tk.NORMAL, text="🔍 Scan All URLs"
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    def load_urls_from_file(self):
        path = filedialog.askopenfilename(
            title="Select URLs File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            self.batch_text.delete("1.0", tk.END)
            self.batch_text.insert("1.0", content)
            cnt = len([u for u in content.splitlines() if u.strip()])
            self.log(f"Loaded {cnt} URLs from file", "SUCCESS")
        except Exception as e:
            self.log(f"Error loading URLs: {e}", "ERROR")
            messagebox.showerror("Error", f"Error loading file:\n{e}")

    # ======================== STATS / EXPORT ========================

    def update_statistics(self):
        total = len(self.scan_results)
        malicious = sum(r.get("is_malicious", False) for r in self.scan_results)
        unknown = sum("error" in r for r in self.scan_results)
        safe = total - malicious - unknown
        self.stats_labels["Total Scanned"].config(text=str(total))
        self.stats_labels["Malicious"].config(text=str(malicious))
        self.stats_labels["Safe"].config(text=str(safe))
        self.stats_labels["Unknown"].config(text=str(unknown))

    def clear_results(self):
        if not self.scan_results:
            return
        if not messagebox.askyesno("Clear Results", "Clear all results?"):
            return
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.scan_results = []
        self.update_statistics()
        self.log("Results cleared", "INFO")

    def export_results(self):
        if not self.scan_results:
            messagebox.showinfo("No Results", "No results to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            pd.DataFrame(self.scan_results).to_csv(path, index=False)
            self.log(f"Results exported to {path}", "SUCCESS")
            messagebox.showinfo("Success", f"Exported to:\n{path}")
        except Exception as e:
            self.log(f"Export failed: {e}", "ERROR")
            messagebox.showerror("Error", f"Export failed:\n{e}")

    def show_statistics(self):
        if not self.scan_results:
            messagebox.showinfo("No Results", "No data yet.")
            return
        total = len(self.scan_results)
        malicious = sum(r.get("is_malicious", False) for r in self.scan_results)
        safe = total - malicious
        confs = [r.get("confidence", 0.0) for r in self.scan_results]
        avg_conf = float(np.mean(confs)) if confs else 0.0
        text = (
            f"Total URLs: {total}\n"
            f"Malicious: {malicious} ({(malicious/total*100 if total else 0):.1f}%)\n"
            f"Safe: {safe} ({(safe/total*100 if total else 0):.1f}%)\n"
            f"Average confidence: {avg_conf:.2%}"
        )
        messagebox.showinfo("Statistics", text)


def main():
    print("=" * 60)
    print("🛡️  Malicious URL Detection System")
    print("=" * 60)
    root = tk.Tk()
    URLScannerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
