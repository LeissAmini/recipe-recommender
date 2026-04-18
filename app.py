from flask import Flask, render_template, request
import json

app = Flask(__name__)


def score_recipe(recipe, pantry, budget, preference):
    score = 0

    pantry_set = set(item.strip().lower() for item in pantry if item.strip())
    recipe_ingredients = set(item.strip().lower() for item in recipe["ingredients"])

    matches = len(recipe_ingredients & pantry_set)
    score += matches * 2

    if recipe["cost"] <= budget:
        score += 5

    if preference != "any" and recipe["category"].lower() == preference.lower():
        score += 3

    if recipe.get("prep_time", 999) <= 20:
        score += 1

    missing_ingredients = list(recipe_ingredients - pantry_set)

    return score, matches, missing_ingredients


def get_top_recipes(pantry, budget, preference):
    with open("recipes.json", "r") as file:
        recipes = json.load(file)

    scored_recipes = []

    for recipe in recipes:
        score, matches, missing_ingredients = score_recipe(recipe, pantry, budget, preference)

        recipe_copy = recipe.copy()
        recipe_copy["score"] = score
        recipe_copy["matches"] = matches
        recipe_copy["missing_ingredients"] = missing_ingredients

        scored_recipes.append(recipe_copy)

    scored_recipes.sort(key=lambda r: (-r["score"], r["cost"]))
    return scored_recipes[:5]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/results", methods=["POST"])
def results():
    budget = float(request.form["budget"])
    pantry = request.form["pantry"].split(",")
    preference = request.form["preference"]

    top_recipes = get_top_recipes(pantry, budget, preference)

    return render_template(
        "results.html",
        recipes=top_recipes,
        budget=budget,
        preference=preference
    )


if __name__ == "__main__":
    app.run(debug=True)