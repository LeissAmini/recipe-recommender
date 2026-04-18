from flask import Flask, render_template, request
import json

app = Flask(__name__)

VALID_PREFERENCES = {"any", "chicken", "vegetarian", "quick"}


def load_recipes():
    with open("recipes.json", "r") as file:
        return json.load(file)


def get_valid_ingredients(recipes):
    ingredients = set()
    for recipe in recipes:
        for ingredient in recipe["ingredients"]:
            ingredients.add(ingredient.strip().lower())
    return sorted(ingredients)


def parse_pantry_input(pantry_raw, valid_ingredients):
    entered = [item.strip().lower() for item in pantry_raw.split(",") if item.strip()]
    valid = [item for item in entered if item in valid_ingredients]
    invalid = [item for item in entered if item not in valid_ingredients]
    return valid, invalid


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

    missing_ingredients = sorted(list(recipe_ingredients - pantry_set))

    return score, matches, missing_ingredients


def get_top_recipes(pantry, budget, preference):
    recipes = load_recipes()
    scored_recipes = []

    for recipe in recipes:
        score, matches, missing_ingredients = score_recipe(recipe, pantry, budget, preference)

        if recipe["cost"] > budget:
            continue

        if preference != "any" and recipe["category"].lower() != preference.lower():
            continue

        if matches == 0:
            continue

        recipe_copy = recipe.copy()
        recipe_copy["score"] = score
        recipe_copy["matches"] = matches
        recipe_copy["missing_ingredients"] = missing_ingredients

        scored_recipes.append(recipe_copy)

    scored_recipes.sort(key=lambda r: (-r["score"], r["cost"], r["name"]))
    return scored_recipes[:5]


@app.route("/", methods=["GET"])
def index():
    recipes = load_recipes()
    valid_ingredients = get_valid_ingredients(recipes)

    return render_template(
        "index.html",
        error=None,
        budget="",
        pantry="",
        preference="any",
        invalid_ingredients=[],
        ingredient_examples=", ".join(valid_ingredients[:10])
    )


@app.route("/results", methods=["POST"])
def results():
    recipes = load_recipes()
    valid_ingredients = get_valid_ingredients(recipes)

    budget_raw = request.form.get("budget", "").strip()
    pantry_raw = request.form.get("pantry", "").strip()
    preference = request.form.get("preference", "any").strip().lower()

    error = None

    if not budget_raw:
        error = "Please enter a budget."
    elif not pantry_raw:
        error = "Please enter at least one pantry ingredient."
    elif preference not in VALID_PREFERENCES:
        error = "Please choose a valid preference."

    budget = None
    if error is None:
        try:
            budget = float(budget_raw)
            if budget <= 0:
                error = "Budget must be greater than 0."
        except ValueError:
            error = "Budget must be a valid number."

    valid_pantry, invalid_ingredients = parse_pantry_input(pantry_raw, valid_ingredients)

    if error is None and len(valid_pantry) == 0:
        error = "Please enter at least one valid ingredient from the list of supported ingredients."

    if error:
        return render_template(
            "index.html",
            error=error,
            budget=budget_raw,
            pantry=pantry_raw,
            preference=preference,
            invalid_ingredients=invalid_ingredients,
            ingredient_examples=", ".join(valid_ingredients[:10])
        )

    top_recipes = get_top_recipes(valid_pantry, budget, preference)

    return render_template(
        "results.html",
        recipes=top_recipes,
        budget=f"{budget:.2f}",
        pantry=", ".join(valid_pantry),
        preference=preference,
        invalid_ingredients=invalid_ingredients
    )


if __name__ == "__main__":
    app.run(debug=True)