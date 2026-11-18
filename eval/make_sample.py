"""Generates a synthetic 4-topic demo transcript (eval/sample_transcript.txt) + eval/sample.json.

It exists only to demo/test the pipeline and evaluation harness offline. For REAL results in your
report, put your own videos (URL + reference summary + questions) in a JSON file like eval/sample.json.
"""
import json
import random

random.seed(7)
TOPICS = {
    "machine learning": ["overfitting", "regularization", "gradient descent", "training data", "validation set",
                          "feature scaling", "decision tree", "random forest", "cross validation", "loss function",
                          "bias variance", "hyperparameters"],
    "databases": ["primary key", "foreign key", "inner join", "index", "normalization", "transaction",
                  "query optimizer", "stored procedure", "group by", "subquery", "schema design", "deadlock"],
    "cloud deployment": ["docker container", "kubernetes cluster", "load balancer", "continuous integration",
                         "environment variables", "autoscaling", "reverse proxy", "deployment pipeline",
                         "health checks", "rolling update", "container registry", "monitoring dashboard"],
    "web security": ["password hashing", "session cookie", "cross site scripting", "sql injection", "rate limiting",
                     "access token", "https certificate", "input validation", "security headers", "two factor login",
                     "csrf protection", "audit logging"],
}
TEMPL = [
    "{a} is an important part of {t} because it directly affects {b}.",
    "When you work with {a} you should also understand {b} and {c}.",
    "A common mistake with {a} is ignoring {b} in a real project.",
    "Let us look at how {a} connects to {b} in practice.",
    "{a} helps with {b}, and {c} makes the whole process faster.",
    "Many beginners confuse {a} with {b}, so we will compare them carefully.",
    "In production {t} teams monitor {a} and review {b} every week.",
]
lines, t = [], 0
for topic, terms in TOPICS.items():
    for _ in range(26):
        a, b, c = random.sample(terms, 3)
        lines.append(f"[{t // 60:02d}:{t % 60:02d}] " + random.choice(TEMPL).format(a=a, b=b, c=c, t=topic).capitalize())
        t += random.randint(9, 15)
open("eval/sample_transcript.txt", "w", encoding="utf-8").write("\n".join(lines))
qa = [
    {"q": "What helps reduce overfitting?", "answer_terms": ["overfitting", "regularization"]},
    {"q": "How do inner joins relate to foreign keys?", "answer_terms": ["inner join", "foreign key"]},
    {"q": "What does a load balancer do in a kubernetes cluster?", "answer_terms": ["load balancer", "kubernetes"]},
    {"q": "Why hash passwords?", "answer_terms": ["password hashing"]},
    {"q": "What is sql injection and input validation?", "answer_terms": ["sql injection", "input validation"]},
    {"q": "Explain gradient descent and the loss function", "answer_terms": ["gradient descent", "loss function"]},
    {"q": "What is database normalization?", "answer_terms": ["normalization"]},
    {"q": "What are rolling updates?", "answer_terms": ["rolling update"]},
]
ref = ("The talk covers machine learning basics such as overfitting and regularization, database design with keys "
       "and joins, cloud deployment with docker and kubernetes, and web security practices like password hashing.")
json.dump([{"name": "synthetic-demo", "transcript_file": "eval/sample_transcript.txt",
            "reference_summary": ref, "qa": qa}], open("eval/sample.json", "w"), indent=2)
print(len(lines), "lines written")
