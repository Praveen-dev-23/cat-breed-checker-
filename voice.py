"""
Text-to-Speech Assistant using FREE models
Options:
1. gTTS (Google Text-to-Speech) - Free, requires internet
2. pyttsx3 (Offline) - Free, no internet needed
"""

import sys
import os

def use_gtts(text, output_file="output.mp3", language='en'):
    """
    Google Text-to-Speech (gTTS) - Completely FREE
    Requires: pip install gtts
    """
    try:
        from gtts import gTTS
    except ImportError:
        print("❌ gTTS not installed. Install it with: pip install gtts")
        return False
    
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(output_file)
        print(f"✅ Saved to {output_file}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def use_pyttsx3(text, output_file="output.wav", rate=150):
    """
    pyttsx3 - Offline Text-to-Speech (COMPLETELY FREE)
    Requires: pip install pyttsx3
    No internet needed!
    """
    try:
        import pyttsx3
    except ImportError:
        print("❌ pyttsx3 not installed. Install it with: pip install pyttsx3")
        return False
    
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)  # Speed of speech
        engine.save_to_file(text, output_file)
        engine.runAndWait()
        print(f"✅ Saved to {output_file}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def interactive_tts():
    """Interactive Text-to-Speech Assistant"""
    print("\n" + "="*50)
    print("🎙️  FREE Text-to-Speech Assistant")
    print("="*50)
    
    print("\nChoose a method:")
    print("1. gTTS (Google) - Free, requires internet")
    print("2. pyttsx3 (Offline) - Free, no internet needed")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    text = input("Enter text to convert: ").strip()
    
    if not text:
        print("❌ No text provided!")
        return
    
    if choice == "1":
        use_gtts(text)
    elif choice == "2":
        use_pyttsx3(text)
    else:
        print("❌ Invalid choice!")


if __name__ == "__main__":
    # If text is provided as command line argument
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(f"Converting: '{text}'")
        print("\nUsing pyttsx3 (offline)...")
        use_pyttsx3(text)
    else:
        # Interactive mode
        interactive_tts()