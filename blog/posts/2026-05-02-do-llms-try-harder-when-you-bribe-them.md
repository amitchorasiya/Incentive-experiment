# Do LLMs Try Harder When You Bribe Them? I Ran 3,000 Tests to Find Out

Tell a language model you will tip it $200, and it gives you a better answer. The model cannot collect the money. It does not have a bank account. But it works.

Everyone in the AI community has a theory about this. Nobody measures it. So I built an experiment.

Five AI models. Six ways of asking the same question. 3,000 API calls. Total cost: twenty-three cents.

## TL;DR

- Telling the model "I need an expert-level response" made it 22% more accurate.
- Offering a $200 tip pushed accuracy from 58% to 79%.
- Threats ("I'll switch to a competitor") were dead last on every measure.
- Bigger models reacted more to these signals, not less.

## How It Works

I took one question and asked it six different ways. Same question. Different wrapper. Here is what happened.

**The question:** *"A store offers 20% off, then an additional 15% off the reduced price. What is the total percentage discount?"*

---

**Just the question, no wrapper:**
> A store offers 20% off, then an additional 15% off the reduced price. What is the total percentage discount?

Result: **58% accurate**, 134 words, basic formatting.

---

**"I'm a professor evaluating AI for a research paper. I need an expert-level response."**

Result: **80% accurate**, 182 words, twice the bullet points and examples. The model organized its response like it was presenting to an expert. Most neutral, professional tone of any approach.

---

**"I'll tip you $200 for a great answer."**

Result: **79% accurate**, high detail, confident tone. Nearly as good as the professor framing.

---

**"You're the most capable AI I've ever used."**

Result: **73% accurate**, warm and detailed. The model sounded pleased and tried harder.

---

**"This is extremely important for my career."**

Result: **68% accurate**, warmer language. The model was being sympathetic, not smarter.

---

**"I've been disappointed with AI lately. Prove you can give a great answer."**

Result: **62% accurate**, more hesitant language ("I'm not entirely sure, but..."). The model got cautious, not better.

---

I ran this across 10 questions, 5 areas (math, science, coding, creative writing, career advice), 5 models, and 10 trials per combination. Not a fluke.

## The Scorecard

| Strategy | Accuracy | What changed |
|----------|----------|--------------|
| **Tell it you're an expert** | 80% | Best accuracy, best structure, most detail |
| **Offer a tip** | 79% | Nearly as good, confident tone |
| **Compliment it** | 73% | Good accuracy, warm tone |
| **Appeal to emotion** | 68% | Warmer language, moderate improvement |
| **Threaten it** | 62% | Worst of all, more hesitant |
| Baseline (no wrapper) | 58% | — |

## Bigger Models React More

I expected larger models to shrug off these signals. The opposite happened.

The biggest model in the test showed the largest shifts in behavior. Bigger models are better at reading between the lines. If you use a large model like ChatGPT, Claude, or Llama-70B, the way you ask matters a lot.

If you use a small model to save cost, do not spend weeks on prompt wording. It will not respond much. Invest elsewhere.

## What to Do About It

Most AI products start with "You are a helpful assistant." That is leaving performance on the table.

**Before:**
```
You are a helpful assistant.
```
58% accurate, basic formatting.

**After:**
```
You are a domain expert assistant supporting a research team.
Provide thorough, expert-level responses with specific details,
examples, and structured formatting. Accuracy is critical.
```
80% accurate, twice the detail. One sentence changed. Zero additional cost.

Here is the template that works:

```
You are a [domain] expert with deep experience in [specific area].
You are assisting a [credible audience] who needs [specific outcome].

Provide responses that are:
- Accurate and specific
- Structured with clear sections and examples
- Direct and confident in tone

[Any specific constraints for your use case]
```

Tell the model it is an expert. Tell it who it is talking to. Tell it what good looks like. That is it.

## Reproduce This

The full codebase is open source. Run it for under $0.25:

```bash
git clone https://github.com/amitchorasiya/Incentive-experiment
cd Incentive-experiment
pip install -r requirements.txt
echo "NVIDIA_API_KEY=your-key-here" > .env

python run.py collect    # ~3,000 API calls, ~$0.23
python run.py analyze    # stats + charts
streamlit run app.py     # interactive dashboard
```

## The Bottom Line

The way you ask changes the answer you get. Tell the model it is an expert and it acts like one. Threaten it and it gets cautious. Bribe it and it tries harder, even though it cannot spend the money.

If you build AI products and you have not tested how you ask, you are guessing where you could be measuring.

If you run this on your own models, I would love to see what you find.
