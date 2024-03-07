import sys
import platform
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QRadioButton, QGroupBox, QHBoxLayout, QSlider, QTextEdit
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from io import BytesIO
import os
import subprocess
from gtts import gTTS
import numpy as np
import speech_recognition as sr
import whisper
import torch
# Les importations de modules externes pour la reconnaissance vocale, synthèse vocale et traitement de l'audio
from groq import Groq
from elevenlabs import generate, play
from openai import OpenAI

# Configuration initiale du script
llm = 'openai'  # Choix du modèle de langage (OpenAI)
tts = 'google'  # Choix de la technologie de synthèse vocale (Google TTS)
model = 'small'  # Choix du modèle Whisper pour la reconnaissance vocale
record_timeout = 5  # Temps maximum d'enregistrement audio en secondes
energy_threshold = 800  # Seuil d'énergie pour le microphone
prompt = "Tu es un assistant français, tes réponses sont courtes, ton nom est Josh."  # Message d'introduction

# Définition des clés API pour les services utilisés
os.environ["OPENAI_API_KEY"] = ""
os.environ["ELEVENLABS_API_KEY"] = ""
os.environ["GROQ_API_KEY"] = ""

messages = []  # Stockage des messages pour le contexte de la conversation

def generate_response_via_groq(input_text):
    """
    Génère une réponse en utilisant l'API Groq.
    """
    global messages
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    messages.append({"role": "user", "content": input_text})
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="mixtral-8x7b-32768",
        temperature=1,
        max_tokens=1024,
        top_p=0.5,
        stop=None,
        stream=False,
    )
    response = chat_completion.choices[0].message.content
    messages.append({"role": "assistant", "content": response})
    return response

def generate_response_via_openai(text):
    """
    Génère une réponse en utilisant l'API OpenAI.
    """
    global messages
    openai = OpenAI()
    messages.append({"role": "user", "content": text})
    chat_completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    response = chat_completion.choices[0].message.content
    messages.append({"role": "assistant", "content": response})
    return response

def find_last_punctuation(text, punctuations, eleven_labs_max_length):
    last_index = -1
    for punctuation in punctuations:
        index = text.rfind(punctuation, 0, eleven_labs_max_length)
        if index > last_index:
            last_index = index
    return last_index

def tts_elevenlabs(text, eleven_labs_max_length):
    """
    Utilise ElevenLabs pour la synthèse vocale.
    """
    punctuations = ['.', '?', '!']
    if text[-1] not in punctuations:
        text += "."
    if len(text) > eleven_labs_max_length:
        end_index = find_last_punctuation(text, punctuations, eleven_labs_max_length)
        text = text[:end_index + 1] if end_index != -1 else text[:eleven_labs_max_length]
    audio = generate(
        api_key=os.environ.get("ELEVENLABS_API_KEY"),
        voice="Josh",
        model="eleven_multilingual_v1",
        text=text,
        stream=False
    )
    play(audio)

def tts_gtts(text):
    """
    Utilise gTTS et ffmpeg pour la synthèse vocale.
    """
    mp3_fp = BytesIO()
    tts = gTTS(text=text, lang='fr')
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    command = ['ffplay', '-nodisp', '-autoexit', '-i', '-']
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    process.stdin.write(mp3_fp.getvalue())
    process.stdin.close()
    process.wait()

def tts_mac(text):
    """
    Utilise la fonctionnalité 'say' de macOS pour la synthèse vocale.
    """
    subprocess.run(['say', text], capture_output=True, text=True)

class WorkerThread(QThread):
    """
    Thread de travail pour la reconnaissance vocale en arrière-plan.
    """
    update_transcription = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self):
        super(WorkerThread, self).__init__()
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.audio_model = whisper.load_model(model, device=DEVICE)


    def stop(self):
        """
        Arrête l'écoute en arrière-plan.
        """
        if hasattr(self, 'stop_listening'):
            self.stop_listening(wait_for_stop=False)
            self.finished.emit()

    def run(self):
        """
        Exécute la reconnaissance vocale en continu jusqu'à l'arrêt du thread.
        """
        recognizer = sr.Recognizer()
        microphone = sr.Microphone(sample_rate=16000)

        with microphone as source:
            recognizer.energy_threshold = energy_threshold
            recognizer.adjust_for_ambient_noise(source)

        def record_callback(_, audio: sr.AudioData) -> None:
                global tts, llm
                try:
                    data = audio.get_raw_data()
                    audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    result = self.audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                    text = result['text'].strip()
                    self.update_transcription.emit(f"--- {text}")
                    response = ''
                    if text.strip():
                        if llm == "openai":
                            response = generate_response_via_openai(text)
                        else:
                            response = generate_response_via_groq(text)

                    self.update_transcription.emit(f"<<< {response}")
                    if response.strip():
                        if tts == "elevenlabs":
                            tts_elevenlabs(response, 90)
                        elif(tts == "mac"):
                            tts_mac(response)
                        else:
                            tts_gtts(response)
                
                except Exception as e:
                    error_message = f"Erreur: {str(e)}"
                    self.update_transcription.emit(error_message)

        self.stop_listening = recognizer.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
        self.update_transcription.emit("Parlez...")

# La classe MainWindow et d'autres définitions de l'interface utilisateur suivent ici...
# Le code de l'interface utilisateur inclut la configuration des widgets, les gestionnaires d'événements et la logique pour démarrer/arrêter l'écoute.
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Assistant Vocal")
        self.setGeometry(100, 100, 800, 600)

        self.initUI()

        self.worker_thread = WorkerThread() # Passe ici les arguments nécessaires
        self.worker_thread.update_transcription.connect(self.updateTranscriptionText)

    def initUI(self):
        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.layout = QVBoxLayout()
        
        self.setupArgumentWidgets()

        self.startButton = QPushButton("Démarrer l'écoute")
        self.startButton.clicked.connect(self.startListening)
        self.layout.addWidget(self.startButton)
        
        self.stopButton = QPushButton("Arrêter l'écoute")
        self.stopButton.clicked.connect(self.stopListening)
        self.stopButton.setEnabled(False)
        self.layout.addWidget(self.stopButton)
        

        self.transcriptionText = QTextEdit()
        self.transcriptionText.setReadOnly(True)
        fontMetrics = self.transcriptionText.fontMetrics()
        lineHeight = fontMetrics.lineSpacing()

        # Estimation pour deux lignes de texte, ajustez le multiplicateur si nécessaire
        heightForTwoLines = lineHeight * 10 + self.transcriptionText.contentsMargins().top() + self.transcriptionText.contentsMargins().bottom()

        self.transcriptionText.setFixedHeight(heightForTwoLines)
        self.layout.addWidget(self.transcriptionText)

        self.centralWidget.setLayout(self.layout)

    def setupArgumentWidgets(self):

        # Dans votre méthode initUI ou setup
        self.cudaGroup = QGroupBox()
        cudaLayout = QHBoxLayout()
        self.cudaStatusLabel = QLabel("CUDA Disponible: Non")
        cudaLayout.addWidget(self.cudaStatusLabel)

        # Mettre à jour le texte selon la disponibilité de CUDA
        if torch.cuda.is_available():
            self.cudaStatusLabel.setText("CUDA Disponible: Oui")
        else:
            self.cudaStatusLabel.setText("CUDA Disponible: Non")
        
        self.cudaGroup.setLayout(cudaLayout)
        self.layout.addWidget(self.cudaGroup)

        # Modèle
        self.modelGroup = QGroupBox("Reconnaissance vocale / Modèle Whisper OpenAi")
        modelLayout = QHBoxLayout()
        self.modelTinyRadio = QRadioButton("Tiny")
        self.modelBaseRadio = QRadioButton("Base")
        self.modelSmallRadio = QRadioButton("Small")
        self.modelMediumRadio = QRadioButton("Medium")
        self.modelLargeRadio = QRadioButton("Large")
        self.modelSmallRadio.setChecked(True)  # Par défaut
        modelLayout.addWidget(self.modelTinyRadio)
        modelLayout.addWidget(self.modelBaseRadio)
        modelLayout.addWidget(self.modelSmallRadio)
        modelLayout.addWidget(self.modelMediumRadio)
        modelLayout.addWidget(self.modelLargeRadio)
        self.modelGroup.setLayout(modelLayout)
        self.layout.addWidget(self.modelGroup)

        # meme ligne horizontal
        horizontalLayout = QHBoxLayout()

        # TTS
        self.ttsGroup = QGroupBox("Synthèse vocale")
        ttsLayout = QHBoxLayout()
        self.ttsGoogleRadio = QRadioButton("Google")
        self.ttsElevenLabsRadio = QRadioButton("Eleven Labs")
        self.ttsMacRadio = QRadioButton("Mac")
        self.ttsGoogleRadio.setChecked(True)  # Par défaut

        # Ajoutez les boutons radio au layout
        ttsLayout.addWidget(self.ttsGoogleRadio)
        ttsLayout.addWidget(self.ttsElevenLabsRadio)
        ttsLayout.addWidget(self.ttsMacRadio)

        # Vérifiez le système d'exploitation et désactivez le bouton radio Mac si nécessaire
        if platform.system() != "Darwin":  # "Darwin" est le nom du système pour macOS
            self.ttsMacRadio.setEnabled(False)  # Rend le bouton radio Mac inactif/grisé

        self.ttsGroup.setLayout(ttsLayout)

        # LLM
        self.llmGroup = QGroupBox("Modèle de langage")
        llmLayout = QHBoxLayout()
        self.llmOpenAIRadio = QRadioButton("OpenAI")
        self.llmGroqRadio = QRadioButton("Groq")
        self.llmOpenAIRadio.setChecked(True)  # Par défaut
        llmLayout.addWidget(self.llmOpenAIRadio)
        llmLayout.addWidget(self.llmGroqRadio)
        self.llmGroup.setLayout(llmLayout)

        # on aligne
        # Ajoutez les deux groupes au layout horizontal
        horizontalLayout.addWidget(self.llmGroup)
        horizontalLayout.addWidget(self.ttsGroup)

        # Enfin, ajoutez le layout horizontal à votre layout principal
        self.layout.addLayout(horizontalLayout)


        # Créez un layout horizontal pour aligner 'energyThresholdGroup' et 'recordTimeoutGroup' sur la même ligne
        horizontalLayoutForThresholdAndTimeout = QHBoxLayout()

        # Seuil d'énergie
        self.energyThresholdGroup = QGroupBox("Seuil d'énergie de détection du micro.")
        energyThresholdLayout = QVBoxLayout()  # Utilisez QVBoxLayout pour le slider et le label

        # Configuration du slider pour le seuil d'énergie
        self.energyThresholdSlider = QSlider(Qt.Horizontal)
        self.energyThresholdSlider.setMinimum(0)
        self.energyThresholdSlider.setMaximum(1500)
        self.energyThresholdSlider.setValue(800)
        self.energyThresholdSlider.setTickPosition(QSlider.TicksBelow)
        self.energyThresholdSlider.setTickInterval(150)
        self.energyThresholdSlider.setSingleStep(1)

        # Label pour le seuil d'énergie
        self.energyThresholdLabel = QLabel("Seuil d'énergie : 800")
        self.energyThresholdSlider.valueChanged.connect(self.updateEnergyThresholdLabel)

        # Ajoutez le slider et le label au layout du groupe
        energyThresholdLayout.addWidget(self.energyThresholdSlider)
        energyThresholdLayout.addWidget(self.energyThresholdLabel)
        self.energyThresholdGroup.setLayout(energyThresholdLayout)

        # Durée d'enregistrement
        self.recordTimeoutGroup = QGroupBox("Durée d'enregistrement (secondes)")
        recordTimeoutLayout = QVBoxLayout()

        # Configuration du slider pour la durée d'enregistrement
        self.recordTimeoutSlider = QSlider(Qt.Horizontal)
        self.recordTimeoutSlider.setMinimum(0)
        self.recordTimeoutSlider.setMaximum(30)
        self.recordTimeoutSlider.setValue(5)
        self.recordTimeoutSlider.setTickPosition(QSlider.TicksBelow)
        self.recordTimeoutSlider.setTickInterval(5)
        self.recordTimeoutSlider.setSingleStep(1)

        # Label pour la durée d'enregistrement
        self.recordTimeoutLabel = QLabel("Durée d'enregistrement : 5 secondes")
        self.recordTimeoutSlider.valueChanged.connect(self.updateRecordTimeoutLabel)

        # Ajoutez le slider et le label au layout du groupe
        recordTimeoutLayout.addWidget(self.recordTimeoutSlider)
        recordTimeoutLayout.addWidget(self.recordTimeoutLabel)
        self.recordTimeoutGroup.setLayout(recordTimeoutLayout)

        # Ajoutez les deux groupes au layout horizontal
        horizontalLayoutForThresholdAndTimeout.addWidget(self.energyThresholdGroup)
        horizontalLayoutForThresholdAndTimeout.addWidget(self.recordTimeoutGroup)

        # Enfin, ajoutez le layout horizontal à votre layout principal
        self.layout.addLayout(horizontalLayoutForThresholdAndTimeout)

        # Création du QLabel pour le titre
        self.promptTitle = QLabel("Pré-prompt (instructions) :")
        self.promptTitle.setAlignment(Qt.AlignLeft)  # Left le texte du titre

        # Ajout du QLabel au layout avant le QTextEdit
        self.layout.addWidget(self.promptTitle)

        # Ton QTextEdit existant
        self.promptText = QTextEdit(prompt)
        self.promptText.setReadOnly(False)
        fontMetrics = self.promptText.fontMetrics()
        lineHeight = fontMetrics.lineSpacing()

        # Estimation pour deux lignes de texte, ajuste le multiplicateur si nécessaire
        heightForTwoLines = lineHeight * 3 + self.promptText.contentsMargins().top() + self.promptText.contentsMargins().bottom()

        self.promptText.setFixedHeight(heightForTwoLines)
        self.layout.addWidget(self.promptText)


    def updateEnergyThresholdLabel(self, value):
        self.energyThresholdLabel.setText(f"Seuil d'énergie : {value}")

    def updateRecordTimeoutLabel(self, value):
        self.recordTimeoutLabel.setText(f"Durée d'enregistrement : {value} secondes")


    def startListening(self):
        global llm, tts, model, energy_threshold, record_timeout, prompt, messages
        # Initialisation de la session de messages
        messages = [{
            "role": "system",
            "content": prompt
        }]

        # Vider le QTextEdit avant de commencer l'écoute
        self.transcriptionText.clear()

         # Extrait les valeurs sélectionnées pour llm, tts, et model
        llm = 'openai' if self.llmOpenAIRadio.isChecked() else 'groq'

        if self.ttsGoogleRadio.isChecked():
            tts = 'google'
        elif(self.ttsElevenLabsRadio.isChecked()):
            tts = 'elevenlabs'
        elif(self.ttsMacRadio.isChecked()):
            tts = 'mac'
        else:
            tts = 'google'

        if self.modelTinyRadio.isChecked():
            model = 'tiny'
        elif(self.modelBaseRadio.isChecked()):
            model = 'base'
        elif(self.modelSmallRadio.isChecked()):
            model = 'small'
        elif(self.modelMediumRadio.isChecked()):
            model = 'medium'
        elif(self.modelLargeRadio.isChecked()):
            model = 'large'
        else:
            model = 'base'

        energy_threshold = self.energyThresholdSlider.value()
        record_timeout = self.recordTimeoutSlider.value()

        prompt = self.promptText.toPlainText()

        print(f"energy_threshold {energy_threshold}  - record_timeout {record_timeout}")

        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        # Configure et démarre le thread worker ici
        # Créez le thread avec les paramètres sélectionnés
        self.worker_thread = WorkerThread()

        self.worker_thread.update_transcription.connect(self.updateTranscriptionText)
        self.worker_thread.finished.connect(self.onListeningFinished)  # Gérez la fin de l'écoute
        self.worker_thread.start()
        
    def stopListening(self):
        self.worker_thread.stop()
        self.stopButton.setEnabled(False)

    def onListeningFinished(self):
        # Cette méthode peut être utilisée pour nettoyer ou réinitialiser l'état après l'arrêt de l'écoute
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)

    def updateTranscriptionText(self, text):
        self.transcriptionText.append(text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
