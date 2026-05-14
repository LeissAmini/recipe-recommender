#Includes all three modes: pantry matching, budget filtering, and cuisine/diet preferences. Ran a few test scenarios and the results are looking pretty good
"""
Budget-Aware Recipe Recommendation System
CPSC 481 - Artificial Intelligence
Using Bayesian Networks (pgmpy)

This prototype demonstrates the full pipeline:

recipe_recommender.py
17 KB
Here's where we're at and what's needed next:

 I'll need an API endpoint (Flask or FastAPI works) to wrap the recommendation function so the frontend can call it. 

Hey guys, just finished vibe coding the first working prototype of the Bayesian Network recommendation engine. It's running with pgmpy and already handles all three modes: pantry matching, budget filtering, and cuisine/diet preferences. Ran a few test scenarios and the results are looking pretty good
"""

'''
Budget-Aware Recipe Recommendation System
CPSC 481 - Artificial Intelligence
Using Bayesian Networks (pgmpy)

This prototype demonstrates the full pipeline:

recipe_recommender.py


PROJECT PROPOSAL: BUDGET-AWARE RECIPE RECOMMENDATION SYSTEM 
Team members: Diana Maldonado CWID: 839805504 di.maldonado5504@csu.fullerton.edu 
Reign Pierson CWID: 887962868 charles.pierson5@csu.fullerton.edu 
Leiss Amini CWID: 840432264 Lamini3@csu.fullerton.edu 
'''



"""
Budget-Aware Recipe Recommendation System
CPSC 481 - Artificial Intelligence
Using Bayesian Networks (pgmpy)

This prototype demonstrates the full pipeline:
1. Define Bayesian Network structure
2. Define Conditional Probability Tables (CPTs)
3. Load recipe data
4. Perform inference given user inputs
5. Rank and return recipe recommendations
"""

import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# ──────────────────────────────────────────────
# 1. RECIPE DATABASE
# ──────────────────────────────────────────────
RECIPES = [
    {
        "id": 1,
        "name": "Chicken Stir Fry",
        "cuisine": "asian",
        "diet": "none",
        "cost": "low",          # low=$0-8, medium=$8-15, high=$15+
        "cook_time": "short",   # short=<30min, medium=30-60, long=>60
        "ingredients": ["chicken", "rice", "soy sauce", "broccoli", "garlic", "oil"],
        "price_estimate": 7.50,
    },
    {
        "id": 2,
        "name": "Spaghetti Bolognese",
        "cuisine": "italian",
        "diet": "none",
        "cost": "low",
        "cook_time": "medium",
        "ingredients": ["ground beef", "pasta", "tomato sauce", "onion", "garlic", "olive oil"],
        "price_estimate": 9.00,
    },
    {
        "id": 3,
        "name": "Veggie Buddha Bowl",
        "cuisine": "american",
        "diet": "vegetarian",
        "cost": "medium",
        "cook_time": "short",
        "ingredients": ["quinoa", "chickpeas", "avocado", "spinach", "sweet potato", "tahini"],
        "price_estimate": 11.00,
    },
    {
        "id": 4,
        "name": "Beef Tacos",
        "cuisine": "mexican",
        "diet": "none",
        "cost": "low",
        "cook_time": "short",
        "ingredients": ["ground beef", "tortillas", "lettuce", "cheese", "salsa", "onion"],
        "price_estimate": 8.00,
    },
    {
        "id": 5,
        "name": "Salmon Teriyaki",
        "cuisine": "asian",
        "diet": "none",
        "cost": "high",
        "cook_time": "medium",
        "ingredients": ["salmon", "rice", "soy sauce", "ginger", "sesame oil", "broccoli"],
        "price_estimate": 16.00,
    },
    {
        "id": 6,
        "name": "Margherita Pizza (Homemade)",
        "cuisine": "italian",
        "diet": "vegetarian",
        "cost": "low",
        "cook_time": "medium",
        "ingredients": ["flour", "mozzarella", "tomato sauce", "basil", "olive oil", "yeast"],
        "price_estimate": 6.50,
    },
    {
        "id": 7,
        "name": "Shrimp Alfredo",
        "cuisine": "italian",
        "diet": "none",
        "cost": "high",
        "cook_time": "medium",
        "ingredients": ["shrimp", "pasta", "heavy cream", "parmesan", "garlic", "butter"],
        "price_estimate": 15.50,
    },
    {
        "id": 8,
        "name": "Black Bean Burritos",
        "cuisine": "mexican",
        "diet": "vegetarian",
        "cost": "low",
        "cook_time": "short",
        "ingredients": ["black beans", "tortillas", "rice", "cheese", "salsa", "sour cream"],
        "price_estimate": 5.50,
    },
    {
        "id": 9,
        "name": "Grilled Chicken Salad",
        "cuisine": "american",
        "diet": "none",
        "cost": "medium",
        "cook_time": "short",
        "ingredients": ["chicken", "lettuce", "tomato", "cucumber", "olive oil", "lemon"],
        "price_estimate": 10.00,
    },
    {
        "id": 10,
        "name": "Pad Thai",
        "cuisine": "asian",
        "diet": "none",
        "cost": "medium",
        "cook_time": "short",
        "ingredients": ["rice noodles", "shrimp", "egg", "peanuts", "bean sprouts", "lime"],
        "price_estimate": 12.00,
    },
    {
        "id": 11,
        "name": "Egg Fried Rice",
        "cuisine": "asian",
        "diet": "none",
        "cost": "low",
        "cook_time": "short",
        "ingredients": ["rice", "egg", "soy sauce", "green onion", "oil", "garlic"],
        "price_estimate": 5.00,
    },
    {
        "id": 12,
        "name": "Chicken Tacos",
        "cuisine": "mexican",
        "diet": "none",
        "cost": "medium",
        "cook_time": "short",
        "ingredients": ["chicken", "tortillas", "salsa", "cheese", "lettuce", "lime"],
        "price_estimate": 8.50,
    },
    {
        "id": 13,
        "name": "Veggie Pasta",
        "cuisine": "italian",
        "diet": "vegetarian",
        "cost": "low",
        "cook_time": "short",
        "ingredients": ["pasta", "tomato sauce", "bell pepper", "mushroom", "garlic", "olive oil"],
        "price_estimate": 7.00,
    },
    {
        "id": 14,
        "name": "Caesar Salad",
        "cuisine": "american",
        "diet": "vegetarian",
        "cost": "low",
        "cook_time": "short",
        "ingredients": ["lettuce", "parmesan", "croutons", "lemon", "garlic", "olive oil"],
        "price_estimate": 6.50,
    },
    {
        "id": 15,
        "name": "Chicken Noodle Soup",
        "cuisine": "american",
        "diet": "none",
        "cost": "medium",
        "cook_time": "medium",
        "ingredients": ["chicken", "carrot", "celery", "onion", "garlic", "noodles"],
        "price_estimate": 9.00,
    },
    {
        "id": 16,
        "name": "Shrimp Fried Rice",
        "cuisine": "asian",
        "diet": "none",
        "cost": "medium",
        "cook_time": "short",
        "ingredients": ["shrimp", "rice", "egg", "soy sauce", "green onion", "oil"],
        "price_estimate": 11.00,
    },
    {
        "id": 17,
        "name": "Black Bean Soup",
        "cuisine": "mexican",
        "diet": "vegetarian",
        "cost": "low",
        "cook_time": "medium",
        "ingredients": ["black beans", "onion", "garlic", "tomato", "cumin", "broth"],
        "price_estimate": 5.50,
    },
    {
        "id": 18,
        "name": "Mushroom Risotto",
        "cuisine": "italian",
        "diet": "vegetarian",
        "cost": "medium",
        "cook_time": "medium",
        "ingredients": ["rice", "mushroom", "parmesan", "onion", "garlic", "butter"],
        "price_estimate": 12.00,
    },
]

# ──────────────────────────────────────────────
# 2. BAYESIAN NETWORK STRUCTURE
# ──────────────────────────────────────────────
# Nodes and their discrete states:
#   CuisinePref:  asian, italian, mexican, american
#   DietPref:     none, vegetarian
#   Budget:       low, medium, high
#   CookTime:     short, medium, long
#   PantryMatch:  low, medium, high  (how many ingredients the user already has)
#   Recommend:    yes, no            (target variable)
#
# Edges (causal relationships):
#   CuisinePref  -> Recommend
#   DietPref     -> Recommend
#   Budget       -> Recommend
#   CookTime     -> Recommend
#   PantryMatch  -> Recommend

def build_network():
    """Build and return the Bayesian Network with CPTs."""

    model = BayesianNetwork([
        ("CuisinePref", "Recommend"),
        ("DietPref", "Recommend"),
        ("Budget", "Recommend"),
        ("CookTime", "Recommend"),
        ("PantryMatch", "Recommend"),
    ])

    # --- Prior CPDs (uniform priors — will be overridden by evidence) ---

    cpd_cuisine = TabularCPD(
        variable="CuisinePref",
        variable_card=4,
        values=[[0.25], [0.25], [0.25], [0.25]],
        state_names={"CuisinePref": ["asian", "italian", "mexican", "american"]},
    )

    cpd_diet = TabularCPD(
        variable="DietPref",
        variable_card=2,
        values=[[0.7], [0.3]],
        state_names={"DietPref": ["none", "vegetarian"]},
    )

    cpd_budget = TabularCPD(
        variable="Budget",
        variable_card=3,
        values=[[0.4], [0.35], [0.25]],
        state_names={"Budget": ["low", "medium", "high"]},
    )

    cpd_cooktime = TabularCPD(
        variable="CookTime",
        variable_card=3,
        values=[[0.4], [0.4], [0.2]],
        state_names={"CookTime": ["short", "medium", "long"]},
    )

    cpd_pantry = TabularCPD(
        variable="PantryMatch",
        variable_card=3,
        values=[[0.33], [0.34], [0.33]],
        state_names={"PantryMatch": ["low", "medium", "high"]},
    )

    # --- Conditional CPD for Recommend ---
    # P(Recommend | CuisinePref, DietPref, Budget, CookTime, PantryMatch)
    #
    # Total parent combinations: 4 * 2 * 3 * 3 * 3 = 216
    # We'll generate this programmatically using a scoring heuristic,
    # then normalize to valid probabilities.

    import itertools

    cuisine_states = ["asian", "italian", "mexican", "american"]
    diet_states = ["none", "vegetarian"]
    budget_states = ["low", "medium", "high"]
    cooktime_states = ["short", "medium", "long"]
    pantry_states = ["low", "medium", "high"]

    yes_probs = []
    no_probs = []

    for cuisine, diet, budget, cooktime, pantry in itertools.product(
        cuisine_states, diet_states, budget_states, cooktime_states, pantry_states
    ):
        # Base score — each factor contributes to likelihood of recommendation
        score = 0.5  # baseline

        # Budget alignment: low budget users prefer low-cost recipes
        if budget == "low":
            score += 0.15
        elif budget == "medium":
            score += 0.10

        # Cook time: shorter is generally preferred
        if cooktime == "short":
            score += 0.10
        elif cooktime == "long":
            score -= 0.05

        # Pantry match: higher match = stronger recommendation
        if pantry == "high":
            score += 0.20
        elif pantry == "medium":
            score += 0.10
        elif pantry == "low":
            score -= 0.05

        # Diet match adds a boost (vegetarian recipes for vegetarian users)
        if diet == "vegetarian":
            score += 0.05

        # Clamp to [0.05, 0.95]
        score = max(0.05, min(0.95, score))

        yes_probs.append(score)
        no_probs.append(1.0 - score)

    cpd_recommend = TabularCPD(
        variable="Recommend",
        variable_card=2,
        values=[yes_probs, no_probs],  # Row 0 = yes, Row 1 = no
        evidence=["CuisinePref", "DietPref", "Budget", "CookTime", "PantryMatch"],
        evidence_card=[4, 2, 3, 3, 3],
        state_names={
            "Recommend": ["yes", "no"],
            "CuisinePref": cuisine_states,
            "DietPref": diet_states,
            "Budget": budget_states,
            "CookTime": cooktime_states,
            "PantryMatch": pantry_states,
        },
    )

    model.add_cpds(cpd_cuisine, cpd_diet, cpd_budget, cpd_cooktime, cpd_pantry, cpd_recommend)
    assert model.check_model(), "Model validation failed!"
    return model


# ──────────────────────────────────────────────
# 3. PANTRY MATCHING
# ──────────────────────────────────────────────
def compute_pantry_match(user_pantry: list[str], recipe_ingredients: list[str]) -> str:
    """Return 'low', 'medium', or 'high' based on ingredient overlap."""
    if not recipe_ingredients:
        return "low"
    user_set = {i.lower().strip() for i in user_pantry}
    recipe_set = {i.lower().strip() for i in recipe_ingredients}
    overlap = len(user_set & recipe_set) / len(recipe_set)
    if overlap >= 0.5:
        return "high"
    elif overlap >= 0.25:
        return "medium"
    else:
        return "low"


# ──────────────────────────────────────────────
# 4. BUDGET FILTERING
# ──────────────────────────────────────────────
def get_budget_level(user_budget: float) -> str:
    if user_budget <= 8:
        return "low"
    elif user_budget <= 15:
        return "medium"
    else:
        return "high"


# ──────────────────────────────────────────────
# 5. RECOMMENDATION ENGINE
# ──────────────────────────────────────────────
def recommend_recipes(
    model,
    user_pantry: list[str],
    user_budget: float,
    cuisine_pref: str | None = None,
    diet_pref: str = "none",
    cook_time_pref: str | None = None,
    top_n: int = 5,
):
    """
    Score each recipe using Bayesian inference and return ranked results.

    Parameters:
        model:           trained BayesianNetwork
        user_pantry:     list of ingredient strings the user has
        user_budget:     max dollar amount per meal
        cuisine_pref:    preferred cuisine or None for any
        diet_pref:       "none" or "vegetarian"
        cook_time_pref:  "short", "medium", "long", or None for any
        top_n:           number of recommendations to return
    """
    inference = VariableElimination(model)
    budget_level = get_budget_level(user_budget)

    scored_recipes = []

    for recipe in RECIPES:
        # Filter: skip recipes over budget
        if recipe["price_estimate"] > user_budget:
            continue

        # Filter: diet compatibility
        if diet_pref == "vegetarian" and recipe["diet"] != "vegetarian":
            continue

        # Build evidence dict for this recipe
        pantry_match = compute_pantry_match(user_pantry, recipe["ingredients"])

        evidence = {
            "Budget": budget_level,
            "PantryMatch": pantry_match,
            "DietPref": diet_pref,
        }
        if cuisine_pref:
            evidence["CuisinePref"] = cuisine_pref
        if cook_time_pref:
            evidence["CookTime"] = cook_time_pref

        # Query P(Recommend=yes | evidence)
        result = inference.query(variables=["Recommend"], evidence=evidence)
        prob_yes = result.values[0]  # index 0 = "yes"

        # Bonus: pantry utilization score (ingredients user already has)
        user_set = {i.lower() for i in user_pantry}
        recipe_set = {i.lower() for i in recipe["ingredients"]}
        pantry_overlap = len(user_set & recipe_set)
        missing = recipe_set - user_set
        additional_cost = recipe["price_estimate"] * (len(missing) / max(len(recipe_set), 1))

        scored_recipes.append({
            "recipe": recipe["name"],
            "cuisine": recipe["cuisine"],
            "bn_score": round(prob_yes, 4),
            "price_estimate": recipe["price_estimate"],
            "additional_cost": round(additional_cost, 2),
            "pantry_match": pantry_match,
            "ingredients_owned": pantry_overlap,
            "ingredients_needed": list(missing),
            "cook_time": recipe["cook_time"],
        })

    # Sort by BN score (descending), then by additional cost (ascending)
    scored_recipes.sort(key=lambda r: (-r["bn_score"], r["additional_cost"]))
    return scored_recipes[:top_n]


# ──────────────────────────────────────────────
# 6. DEMO / MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Budget-Aware Recipe Recommendation System")
    print("  CPSC 481 — Bayesian Network Prototype")
    print("=" * 60)

    # Build the model
    print("\n[1] Building Bayesian Network...")
    model = build_network()
    print("    ✓ Network built and validated")
    print(f"    Nodes:  {model.nodes()}")
    print(f"    Edges:  {model.edges()}")

    # --- Scenario 1: Pantry Mode ---
    print("\n" + "─" * 60)
    print("  SCENARIO 1: PANTRY MODE")
    print("─" * 60)
    pantry = ["chicken", "rice", "soy sauce", "garlic", "oil", "tortillas", "cheese", "salsa"]
    budget = 15.00
    print(f"  Pantry:  {pantry}")
    print(f"  Budget:  ${budget:.2f}")
    print(f"  Diet:    any")
    print(f"  Cuisine: any")

    results = recommend_recipes(model, pantry, budget)
    print(f"\n  Top {len(results)} Recommendations:\n")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['recipe']} ({r['cuisine']})")
        print(f"     BN Score: {r['bn_score']}  |  Est. Cost: ${r['price_estimate']:.2f}  |  Add'l Cost: ${r['additional_cost']:.2f}")
        print(f"     Pantry Match: {r['pantry_match']}  |  Owned: {r['ingredients_owned']}  |  Need: {r['ingredients_needed']}")
        print()

    # --- Scenario 2: Budget Mode (vegetarian, $10 max) ---
    print("─" * 60)
    print("  SCENARIO 2: BUDGET MODE (Vegetarian, $10 max)")
    print("─" * 60)
    pantry2 = ["rice", "black beans", "tortillas"]
    budget2 = 10.00
    results2 = recommend_recipes(model, pantry2, budget2, diet_pref="vegetarian")
    print(f"  Pantry:  {pantry2}")
    print(f"  Budget:  ${budget2:.2f}")
    print(f"\n  Top {len(results2)} Recommendations:\n")
    for i, r in enumerate(results2, 1):
        print(f"  {i}. {r['recipe']} ({r['cuisine']})")
        print(f"     BN Score: {r['bn_score']}  |  Est. Cost: ${r['price_estimate']:.2f}  |  Add'l Cost: ${r['additional_cost']:.2f}")
        print(f"     Pantry Match: {r['pantry_match']}  |  Need: {r['ingredients_needed']}")
        print()

    # --- Scenario 3: Cuisine preference ---
    print("─" * 60)
    print("  SCENARIO 3: Asian cuisine, short cook time, $20 budget")
    print("─" * 60)
    pantry3 = ["rice", "soy sauce", "garlic", "broccoli"]
    budget3 = 20.00
    results3 = recommend_recipes(model, pantry3, budget3, cuisine_pref="asian", cook_time_pref="short")
    print(f"  Pantry:  {pantry3}")
    print(f"\n  Top {len(results3)} Recommendations:\n")
    for i, r in enumerate(results3, 1):
        print(f"  {i}. {r['recipe']} ({r['cuisine']})")
        print(f"     BN Score: {r['bn_score']}  |  Est. Cost: ${r['price_estimate']:.2f}  |  Add'l Cost: ${r['additional_cost']:.2f}")
        print(f"     Pantry Match: {r['pantry_match']}  |  Need: {r['ingredients_needed']}")
        print()


if __name__ == "__main__":
    main()
#recipe_recommender.py