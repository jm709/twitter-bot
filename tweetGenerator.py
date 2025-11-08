import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
import random


class TwitterBotGenerator:
    def __init__(self, model_path: str = r"..\my_twitter_bot_model"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully!")

    def clean_generated_text(self, text: str) -> str:
        text = text[100:]
        if len(text) > 240:
            text = text[:237] + "..."
        return text

    def generate_tweet(self, prompt: str = "", max_length: int = 80, temperature: float = 0.9):
        bot_prompt = ("You are a twitter bot named ShallowRobert."
                      "You are stupid. Your goal is to talk about current events.")
        # if you input a prompt it will continue from it, else it will use a starter
        if prompt:
            input_text = bot_prompt.strip() + prompt.strip()
        else:
            starters = ["I think", "Just", "Today", "Why", "Can't believe", "So", "Really", "I hate", "My dream",
                        "Burger", "From the depths of my soul", "The demon comes out when", "Daddy's here", "Rawr"
                        "Pizza", "Toronto", "Breaking news", "Fun fact", "Did you know", "AI-generated", "Dreamscape",
                        "Midnight thoughts", "Unpopular opinion", "Plot twist", "What if", "Imagine this","Overheard",
                        "Strange idea", "Tiny detail", "Bold claim", "Daily reminder", "Flashback", "Future vision",
                        "Glitch", "Fragment", "Prompt:", "Generated idea", "Digital ghost", "Experimental", "Soft chaos",
                        "Uncanny moment", "Fictional post", "Thread begins", "Speculation", "Forecast", "Retro memory",
                        "Echoes of", "Going Beast Mode"]
            input_text = bot_prompt.strip() + random.choice(starters)

        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2
            )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self.clean_generated_text(generated_text)

    def generate_headline(self, gloop, prompt: str, temperature: float = 0.7):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2
            )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            #Don't Question My Method. Deletes the tweet prompt stuff
            x = 312 + 19 + gloop + 20

            generated_text = generated_text[x:]

            return generated_text