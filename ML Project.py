{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "574fad4f-2a3a-4b05-b646-10219d0097e6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "Press Enter after typing the text... The quick browb fix jumps over tge lazy diog\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Words per minute: 29.48\n",
      "Accuracy: 100.00%\n"
     ]
    }
   ],
   "source": [
    "import time\n",
    "import random\n",
    "\n",
    "def collect_typing_data(user_input, actual_text):\n",
    "    # Introduce a small delay to simulate typing\n",
    "    time.sleep(1)  # Delay for 1 second before starting\n",
    "\n",
    "    start_time = time.time()\n",
    "    \n",
    "    # Simulate user typing by waiting for some time\n",
    "    # Here, we can just print and wait for input in real scenarios\n",
    "    input(\"Press Enter after typing the text...\")  # Simulate user action\n",
    "    \n",
    "    end_time = time.time()\n",
    "    \n",
    "    words_typed = user_input.split()\n",
    "    words_actual = actual_text.split()\n",
    "    \n",
    "    typing_duration = end_time - start_time  # Time taken in seconds\n",
    "\n",
    "    # Prevent ZeroDivisionError\n",
    "    if typing_duration == 0:\n",
    "        typing_duration = 0.01  # Set a small value to avoid division by zero\n",
    "\n",
    "    words_per_minute = (len(words_typed) / typing_duration) * 60\n",
    "    correct_words = sum([1 for i in range(len(words_typed)) if i < len(words_actual) and words_typed[i] == words_actual[i]])\n",
    "    \n",
    "    accuracy = (correct_words / len(words_actual)) * 100 if len(words_actual) > 0 else 0\n",
    "    \n",
    "    return words_per_minute, accuracy\n",
    "\n",
    "# Simulated session\n",
    "user_input = \"The quick brown fox jumps over the lazy dog\"\n",
    "actual_text = \"The quick brown fox jumps over the lazy dog\"\n",
    "wpm, accuracy = collect_typing_data(user_input, actual_text)\n",
    "\n",
    "print(f\"Words per minute: {wpm:.2f}\")\n",
    "print(f\"Accuracy: {accuracy:.2f}%\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "14b8f1c7-0090-4811-9bc0-a246bea36994",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Session  WPM   Accuracy\n",
      "0        1   59  86.909094\n",
      "1        2   69  96.451655\n",
      "2        3   43  97.467661\n",
      "3        4   63  90.630527\n",
      "4        5   62  99.729162\n",
      "5        6   78  85.332880\n",
      "6        7   59  98.656767\n",
      "7        8   70  94.959056\n",
      "8        9   65  87.172419\n",
      "9       10   52  95.817665\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Simulating data collection for multiple sessions\n",
    "data = {\n",
    "    \"Session\": list(range(1, 11)),\n",
    "    \"WPM\": [random.randint(40, 80) for _ in range(10)],\n",
    "    \"Accuracy\": [random.uniform(85, 100) for _ in range(10)]\n",
    "}\n",
    "\n",
    "df = pd.DataFrame(data)\n",
    "print(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "0fe5074e-6d27-4998-b30d-1c589e85865e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Session  WPM   Accuracy  WPM_Trend  Accuracy_Trend\n",
      "0        1   59  86.909094        NaN             NaN\n",
      "1        2   69  96.451655        NaN             NaN\n",
      "2        3   43  97.467661  57.000000       93.609470\n",
      "3        4   63  90.630527  58.333333       94.849948\n",
      "4        5   62  99.729162  56.000000       95.942450\n",
      "5        6   78  85.332880  67.666667       91.897523\n",
      "6        7   59  98.656767  66.333333       94.572937\n",
      "7        8   70  94.959056  69.000000       92.982901\n",
      "8        9   65  87.172419  64.666667       93.596081\n",
      "9       10   52  95.817665  62.333333       92.649713\n"
     ]
    }
   ],
   "source": [
    "# Adding trend features (moving average)\n",
    "df['WPM_Trend'] = df['WPM'].rolling(window=3).mean()\n",
    "df['Accuracy_Trend'] = df['Accuracy'].rolling(window=3).mean()\n",
    "\n",
    "print(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "74167435-6b92-4efc-99a2-472fe662ea56",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Squared Error: 247.21573940808105\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import mean_squared_error\n",
    "\n",
    "# Prepare data for regression\n",
    "X = df[['Accuracy', 'WPM_Trend', 'Accuracy_Trend']].fillna(0)\n",
    "y = df['WPM']\n",
    "\n",
    "# Train/test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Regression model\n",
    "model = LinearRegression()\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Predict on test data\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "# Evaluate model performance\n",
    "mse = mean_squared_error(y_test, y_pred)\n",
    "print(f\"Mean Squared Error: {mse}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "a32056c6-ae96-433b-ab50-dd0526cf7b32",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Recommended words for practice: ['cat', 'dog', 'sun']\n"
     ]
    }
   ],
   "source": [
    "# Sample word difficulty list (could be expanded)\n",
    "word_difficulty = {\n",
    "    \"easy\": [\"cat\", \"dog\", \"sun\"],\n",
    "    \"medium\": [\"apple\", \"garden\", \"flower\"],\n",
    "    \"hard\": [\"difficult\", \"interesting\", \"phenomenon\"]\n",
    "}\n",
    "\n",
    "def recommend_practice(accuracy, speed):\n",
    "    if accuracy < 90:\n",
    "        return word_difficulty['easy']\n",
    "    elif speed < 60:\n",
    "        return word_difficulty['medium']\n",
    "    else:\n",
    "        return word_difficulty['hard']\n",
    "\n",
    "# Simulated recommendation\n",
    "recommended_words = recommend_practice(accuracy=wpm, speed=accuracy)\n",
    "print(f\"Recommended words for practice: {recommended_words}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "4bd2910b-86aa-4896-b905-65dc739ce6f9",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "E\n",
      "======================================================================\n",
      "ERROR: C:\\Users\\hp\\AppData\\Roaming\\jupyter\\runtime\\kernel-49e1b0db-4bd2-4282-ae50-2937aa921354 (unittest.loader._FailedTest.C:\\Users\\hp\\AppData\\Roaming\\jupyter\\runtime\\kernel-49e1b0db-4bd2-4282-ae50-2937aa921354)\n",
      "----------------------------------------------------------------------\n",
      "AttributeError: module '__main__' has no attribute 'C:\\Users\\hp\\AppData\\Roaming\\jupyter\\runtime\\kernel-49e1b0db-4bd2-4282-ae50-2937aa921354'\n",
      "\n",
      "----------------------------------------------------------------------\n",
      "Ran 1 test in 0.003s\n",
      "\n",
      "FAILED (errors=1)\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "True",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[1;31mSystemExit\u001b[0m\u001b[1;31m:\u001b[0m True\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\hp\\anaconda3\\Lib\\site-packages\\IPython\\core\\interactiveshell.py:3561: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.\n",
      "  warn(\"To exit: use 'exit', 'quit', or Ctrl-D.\", stacklevel=1)\n"
     ]
    }
   ],
   "source": [
    "import unittest\n",
    "\n",
    "class TestTypingSpeedCoach(unittest.TestCase):\n",
    "\n",
    "    def test_collect_typing_data(self):\n",
    "        user_input = \"The quick brown fox\"\n",
    "        actual_text = \"The quick brown fox\"\n",
    "        wpm, accuracy = collect_typing_data(user_input, actual_text)\n",
    "        \n",
    "        # Check that WPM is a float\n",
    "        self.assertIsInstance(wpm, float)\n",
    "        # Check that accuracy is within 0 to 100\n",
    "        self.assertGreaterEqual(accuracy, 0)\n",
    "        self.assertLessEqual(accuracy, 100)\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    unittest.main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "412b318d-c010-42c7-a4cd-c3622d9ab764",
   "metadata": {},
   "outputs": [],
   "source": [
    "def collect_typing_data(user_input, actual_text):\n",
    "    time.sleep(1)  # Delay for typing simulation\n",
    "    start_time = time.time()\n",
    "    \n",
    "    input(\"Press Enter after typing the text...\")  # Simulate user action\n",
    "    end_time = time.time()\n",
    "\n",
    "    words_typed = user_input.split()\n",
    "    words_actual = actual_text.split()\n",
    "    \n",
    "    typing_duration = end_time - start_time\n",
    "    if typing_duration == 0:\n",
    "        typing_duration = 0.01  # Prevent division by zero\n",
    "\n",
    "    words_per_minute = (len(words_typed) / typing_duration) * 60\n",
    "    correct_words = sum([1 for i in range(len(words_typed)) if i < len(words_actual) and words_typed[i] == words_actual[i]])\n",
    "    accuracy = (correct_words / len(words_actual)) * 100 if len(words_actual) > 0 else 0\n",
    "    \n",
    "    # Calculate error rate\n",
    "    error_count = len(words_actual) - correct_words\n",
    "    error_rate = (error_count / len(words_actual)) * 100 if len(words_actual) > 0 else 0\n",
    "\n",
    "    return words_per_minute, accuracy, error_rate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "c654b301-bca5-45cf-bf72-50714c967298",
   "metadata": {},
   "outputs": [],
   "source": [
    "def recommend_practice(accuracy, speed):\n",
    "    if accuracy < 90:\n",
    "        return [\"cat\", \"dog\", \"sun\"]  # Easy words for accuracy improvement\n",
    "    elif speed < 60:\n",
    "        return [\"apple\", \"garden\", \"flower\"]  # Medium words for speed improvement\n",
    "    else:\n",
    "        return [\"difficult\", \"interesting\", \"phenomenon\"]  # Hard words for advanced practice"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "36333e8f-d5a6-4309-be4d-3dcff7a51601",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Welcome to the AI Typing Speed Coach!\n"
     ]
    },
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "Type the following text: 'The quick brown fox jumps over the lazy dog'\n",
      " gg\n",
      "Press Enter after typing the text... gg\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Results:\n",
      "Words per minute: 30.50\n",
      "Accuracy: 0.00%\n",
      "Error Rate: 100.00%\n"
     ]
    },
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "Do you want to try again? (y/n):  n\n"
     ]
    }
   ],
   "source": [
    "def main():\n",
    "    print(\"Welcome to the AI Typing Speed Coach!\")\n",
    "    actual_text = \"The quick brown fox jumps over the lazy dog\"\n",
    "\n",
    "    while True:\n",
    "        user_input = input(f\"Type the following text: '{actual_text}'\\n\")\n",
    "        wpm, accuracy, error_rate = collect_typing_data(user_input, actual_text)\n",
    "\n",
    "        print(f\"\\nResults:\\nWords per minute: {wpm:.2f}\\nAccuracy: {accuracy:.2f}%\\nError Rate: {error_rate:.2f}%\")\n",
    "        if input(\"Do you want to try again? (y/n): \").lower() != 'y':\n",
    "            break\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "7081f687-0af5-4a34-b8e9-8174eb7ae59b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "ERROR: Could not find a version that satisfies the requirement sqlite3 (from versions: none)\n",
      "ERROR: No matching distribution found for sqlite3\n"
     ]
    }
   ],
   "source": [
    "pip install sqlite3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "1b44de94-38af-4010-bacb-a19fb86fe809",
   "metadata": {},
   "outputs": [],
   "source": [
    "import sqlite3\n",
    "\n",
    "def create_db():\n",
    "    conn = sqlite3.connect('typing_speed_coach.db')\n",
    "    c = conn.cursor()\n",
    "    c.execute('''\n",
    "        CREATE TABLE IF NOT EXISTS typing_sessions (\n",
    "            session_id INTEGER PRIMARY KEY AUTOINCREMENT,\n",
    "            wpm REAL,\n",
    "            accuracy REAL,\n",
    "            error_rate REAL\n",
    "        )\n",
    "    ''')\n",
    "    conn.commit()\n",
    "    conn.close()\n",
    "\n",
    "def save_session(wpm, accuracy, error_rate):\n",
    "    conn = sqlite3.connect('typing_speed_coach.db')\n",
    "    c = conn.cursor()\n",
    "    c.execute('''\n",
    "        INSERT INTO typing_sessions (wpm, accuracy, error_rate)\n",
    "        VALUES (?, ?, ?)\n",
    "    ''', (wpm, accuracy, error_rate))\n",
    "    conn.commit()\n",
    "    conn.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "992d1c93-fb5b-44e2-8e25-866fb4e049f2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: Flask in c:\\users\\hp\\anaconda3\\lib\\site-packages (2.2.5)\n",
      "Requirement already satisfied: Werkzeug>=2.2.2 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from Flask) (2.2.3)\n",
      "Requirement already satisfied: Jinja2>=3.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from Flask) (3.1.3)\n",
      "Requirement already satisfied: itsdangerous>=2.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from Flask) (2.0.1)\n",
      "Requirement already satisfied: click>=8.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from Flask) (8.1.7)\n",
      "Requirement already satisfied: colorama in c:\\users\\hp\\anaconda3\\lib\\site-packages (from click>=8.0->Flask) (0.4.6)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from Jinja2>=3.0->Flask) (2.1.3)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install Flask"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "caa9a325-f59d-4665-954a-451716f07646",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on http://127.0.0.1:5000\n",
      "Press CTRL+C to quit\n",
      " * Restarting with watchdog (windowsapi)\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "1",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[1;31mSystemExit\u001b[0m\u001b[1;31m:\u001b[0m 1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\hp\\anaconda3\\Lib\\site-packages\\IPython\\core\\interactiveshell.py:3561: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.\n",
      "  warn(\"To exit: use 'exit', 'quit', or Ctrl-D.\", stacklevel=1)\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, render_template, request\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "@app.route('/', methods=['GET', 'POST'])\n",
    "def index():\n",
    "    if request.method == 'POST':\n",
    "        user_input = request.form['user_input']\n",
    "        actual_text = \"The quick brown fox jumps over the lazy dog\"\n",
    "        wpm, accuracy, error_rate = collect_typing_data(user_input, actual_text)\n",
    "        \n",
    "        return render_template('result.html', wpm=wpm, accuracy=accuracy, error_rate=error_rate)\n",
    "    \n",
    "    return render_template('index.html')\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(debug=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "1f99a4f2-ae43-4ff0-b930-ae8d4deac5af",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: streamlit in c:\\users\\hp\\anaconda3\\lib\\site-packages (1.30.0)\n",
      "Requirement already satisfied: altair<6,>=4.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (5.0.1)\n",
      "Requirement already satisfied: blinker<2,>=1.0.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (1.6.2)\n",
      "Requirement already satisfied: cachetools<6,>=4.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (4.2.2)\n",
      "Requirement already satisfied: click<9,>=7.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (8.1.7)\n",
      "Requirement already satisfied: importlib-metadata<8,>=1.4 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (7.0.1)\n",
      "Requirement already satisfied: numpy<2,>=1.19.3 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (1.26.4)\n",
      "Requirement already satisfied: packaging<24,>=16.8 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (23.1)\n",
      "Requirement already satisfied: pandas<3,>=1.3.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (2.1.4)\n",
      "Requirement already satisfied: pillow<11,>=7.1.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (10.2.0)\n",
      "Requirement already satisfied: protobuf<5,>=3.20 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (3.20.3)\n",
      "Requirement already satisfied: pyarrow>=6.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (14.0.2)\n",
      "Requirement already satisfied: python-dateutil<3,>=2.7.3 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (2.8.2)\n",
      "Requirement already satisfied: requests<3,>=2.27 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (2.31.0)\n",
      "Requirement already satisfied: rich<14,>=10.14.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (13.3.5)\n",
      "Requirement already satisfied: tenacity<9,>=8.1.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (8.2.2)\n",
      "Requirement already satisfied: toml<2,>=0.10.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (0.10.2)\n",
      "Requirement already satisfied: typing-extensions<5,>=4.3.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (4.9.0)\n",
      "Requirement already satisfied: tzlocal<6,>=1.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (2.1)\n",
      "Requirement already satisfied: validators<1,>=0.2 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (0.18.2)\n",
      "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (3.1.37)\n",
      "Requirement already satisfied: pydeck<1,>=0.8.0b4 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (0.8.0)\n",
      "Requirement already satisfied: tornado<7,>=6.0.3 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (6.3.3)\n",
      "Requirement already satisfied: watchdog>=2.1.5 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (2.1.6)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from altair<6,>=4.0->streamlit) (3.1.3)\n",
      "Requirement already satisfied: jsonschema>=3.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from altair<6,>=4.0->streamlit) (4.19.2)\n",
      "Requirement already satisfied: toolz in c:\\users\\hp\\anaconda3\\lib\\site-packages (from altair<6,>=4.0->streamlit) (0.12.0)\n",
      "Requirement already satisfied: colorama in c:\\users\\hp\\anaconda3\\lib\\site-packages (from click<9,>=7.0->streamlit) (0.4.6)\n",
      "Requirement already satisfied: gitdb<5,>=4.0.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.7)\n",
      "Requirement already satisfied: zipp>=0.5 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from importlib-metadata<8,>=1.4->streamlit) (3.17.0)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from pandas<3,>=1.3.0->streamlit) (2023.3.post1)\n",
      "Requirement already satisfied: tzdata>=2022.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from pandas<3,>=1.3.0->streamlit) (2023.3)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from python-dateutil<3,>=2.7.3->streamlit) (1.16.0)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (3.4)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2.0.7)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2024.8.30)\n",
      "Requirement already satisfied: markdown-it-py<3.0.0,>=2.2.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (2.2.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (2.15.1)\n",
      "Requirement already satisfied: decorator>=3.4.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from validators<1,>=0.2->streamlit) (5.1.1)\n",
      "Requirement already satisfied: smmap<5,>=3.0.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.0)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.3)\n",
      "Requirement already satisfied: attrs>=22.2.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (23.1.0)\n",
      "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.7.1)\n",
      "Requirement already satisfied: referencing>=0.28.4 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.30.2)\n",
      "Requirement already satisfied: rpds-py>=0.7.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.10.6)\n",
      "Requirement already satisfied: mdurl~=0.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from markdown-it-py<3.0.0,>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.0)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install streamlit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "bbd9dff7-795e-4d54-a2f5-e4e871f1d2ec",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-10-17 18:33:57.819 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\hp\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import time\n",
    "import random\n",
    "import sqlite3\n",
    "\n",
    "# Function to collect typing data\n",
    "def collect_typing_data(user_input, actual_text):\n",
    "    words_typed = user_input.split()\n",
    "    words_actual = actual_text.split()\n",
    "    \n",
    "    correct_words = sum([1 for i in range(len(words_typed)) if i < len(words_actual) and words_typed[i] == words_actual[i]])\n",
    "    accuracy = (correct_words / len(words_actual)) * 100 if len(words_actual) > 0 else 0\n",
    "    \n",
    "    return len(words_typed), accuracy\n",
    "\n",
    "# Function to save session data to the database\n",
    "def save_session(wpm, accuracy, error_rate):\n",
    "    conn = sqlite3.connect('typing_speed_coach.db')\n",
    "    c = conn.cursor()\n",
    "    c.execute('''\n",
    "        INSERT INTO typing_sessions (wpm, accuracy, error_rate)\n",
    "        VALUES (?, ?, ?)\n",
    "    ''', (wpm, accuracy, error_rate))\n",
    "    conn.commit()\n",
    "    conn.close()\n",
    "\n",
    "# Streamlit UI\n",
    "st.title(\"AI Typing Speed Coach\")\n",
    "st.write(\"Type the following text as fast as you can:\")\n",
    "\n",
    "# Actual text to type\n",
    "actual_text = \"The quick brown fox jumps over the lazy dog\"\n",
    "st.write(f\"**Text:** {actual_text}\")\n",
    "\n",
    "# User input\n",
    "user_input = st.text_input(\"Start typing here:\")\n",
    "\n",
    "if st.button(\"Submit\"):\n",
    "    # Simulate a typing duration\n",
    "    time.sleep(1)  # Simulate a small delay before measuring\n",
    "    typed_words_count, accuracy = collect_typing_data(user_input, actual_text)\n",
    "\n",
    "    # Calculate WPM (Words Per Minute)\n",
    "    typing_duration = 1  # Set to a constant for simulation\n",
    "    wpm = (typed_words_count / typing_duration) * 60\n",
    "\n",
    "    # Save session to database\n",
    "    error_rate = (len(actual_text.split()) - typed_words_count) / len(actual_text.split()) * 100\n",
    "    save_session(wpm, accuracy, error_rate)\n",
    "\n",
    "    # Display results\n",
    "    st.write(f\"**Words per minute:** {wpm:.2f}\")\n",
    "    st.write(f\"**Accuracy:** {accuracy:.2f}%\")\n",
    "    st.write(f\"**Error Rate:** {error_rate:.2f}%\")\n",
    "\n",
    "# Create the database if it doesn't exist\n",
    "def create_db():\n",
    "    conn = sqlite3.connect('typing_speed_coach.db')\n",
    "    c = conn.cursor()\n",
    "    c.execute('''\n",
    "        CREATE TABLE IF NOT EXISTS typing_sessions (\n",
    "            session_id INTEGER PRIMARY KEY AUTOINCREMENT,\n",
    "            wpm REAL,\n",
    "            accuracy REAL,\n",
    "            error_rate REAL\n",
    "        )\n",
    "    ''')\n",
    "    conn.commit()\n",
    "    conn.close()\n",
    "\n",
    "create_db()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "0c879609-c168-4af4-b76b-6c0ff6dacca5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip freeze > requirements.txt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8cb77da5-098c-4441-9774-36c546eca674",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
