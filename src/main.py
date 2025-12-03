# Generate inverted index and other necessary files. Run this script once before running the web app.
from indexer import Indexer
import topic_classifier as tc
import scraper
from pathlib import Path

scraper = scraper.Scraper()

def generate_files(data="./wikipedia_scraped_data.csv", index_file="./inverted_index.pkl", classifier_file="./NB_classifier.pkl"):
    health_queries = [
    "global health statistics",
    "disease prevalence",
    "mental health trends",
    "nutrition data",
    "healthcare access",
    "history of medicine",
    "epidemiology",
    "pharmaceutical industry",
    "medical research breakthroughs",
    "healthcare reform",
    "diet and exercise",
    "vaccination rates",
    "public health initiatives",
    "health disparities",
    "chronic diseases"
] # list of queries for health topic
    environment_queries = [
        "global warming",
        "environmental issues",
        "endangered species"
        "renewable energy",
        "deforestation and reforestation",
        "air quality and pollution",
        "landfills",
        "carbon footprint",
        "environmental laws",
        "marine ecosystems",
        "water conservation",
        "sustainable agriculture",
        "recycling programs",
        "climate change",
        "environmental disasters"
    ] # a list of queries for environment topic
    tech_queries = [
        "technology advancements",
        "self-driving cars",
        "artificial intelligence",
        "history of computing",
        "cybersecurity",
        "wearable technology",
        "vR and AR",
        "space exploration technology",
        "computer programming languages",
        "robotics",
        "internet",
        "blockchain technology",
        "quantum computing",
        "5G technology"
    ]
    economy_queries = [
        "stock market performance",
        "job market trends",
        "inflation",
        "global trade",
        "history of economics",
        "national debt",
        "cryptocurrency",
        "consumer spending habits",
        "entrepreneurship and startups",
        "money laundering",
        "real estate market",
        "gig economy",
        "supply chain management"
    ]
    entertainment_queries = [
        "music industry",
        "online streaming",
        "tv and movies",
        "celebrities and hollywood",
        "video game industry",
        "awards shows",
        "history of entertainment",
        "theater and performing arts",
        "sports entertainment",
        "music festivals",
        "government regulation in the entertainment industry",
        "fan culture",
        "comic books and graphic novels"
    ]
    sports_queries = [
        "major sporting events",
        "sports analytics",
        "history of sports",
        "famous athletes",
        "sports medicine",
        "youth sports trends",
        "sports psychology",
        "extreme sports",
        "sports broadcasting",
        "sports management",
        "esports and gaming",
        "olympic games",
        "paralympics"
    ]
    politics_queries = [
        "elections",
        "international relations",
        "political science",
        "government systems",
        "political ideologies",
        "history of politics",
        "political movements",
        "public policy",
        "political scandals",
        "human rights movements",
        "wars and conflicts",
        "diplomacy",
        "voting behavior"
        "political campaigns"
    ]
    education_queries = [
        "educational disparity",
        "literacy rates",
        "student loan data",
        "history of education",
        "online learning trends",
        "special education",
        "early childhood education",
        "higher education statistics",
        "educational technology",
        "homeschooling",
        "language learning",
        "adult education",
        "vocational training",
        "school safety",
        "STEM education",
        "women in STEM"
    ]
    travel_queries = [
        "top tourist destinations",
        "airline industry data",
        "popular tourist attractions",
        "travel safety statistics",
        "history of travel",
        "travel spending habits",
        "hotel industry data",
        "cruise industry",
        "visas and travel regulations",
        "backpacking trends",
        "historical landmarks",
        "eco-tourism",
        "cultural festivals",
        "adventure travel",
        "travel technology",
        "road trip routes",
        "travel photography",
        "space tourism"
    ]
    food_queries = [
        "crop yield statistics",
        "global food shortage",
        "popular dishes by culture",
        "history of food",
        "culinary techniques",
        "nutrition trends",
        "non-gmo trends",
        "molecular gastronomy",
        "beverages and brewing",
        "dietary habits",
        "baking and pastries",
        "food in different cultures",
        "drugs used in food production",
        "food eaten by astronauts",
        "food preservation methods"
    ]

    all_queries = [
    health_queries,
    environment_queries,
    tech_queries,
    economy_queries,
    entertainment_queries,
    sports_queries,
    politics_queries,
    education_queries,
    travel_queries,
    food_queries
]

            # Check if classifier already fit
    
    fp_data = Path(data)

    if not fp_data.exists():
        scraper.run_scraper(all_queries, save=True)
    else:
        print(f"Scraped data file exists at {data}")

    fp_index = Path(index_file)
    if not fp_index.exists():
        index = Indexer()
        index.run_indexer()
        index._save()
    else:
        index = Indexer.load(index_file)

    fp_classifier = Path(classifier_file)
    if not fp_classifier.exists():
        classifier = tc.TopicClassifier()
        classifier._fit()
        classifier._save()
    else:
        classifier = tc.TopicClassifier.load(fp_classifier)

    print("All necessary files generated.")

if __name__ == "__main__":
    generate_files()