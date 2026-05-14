from flask import Flask, render_template, request

from recipe_recommender import RECIPES, build_network, recommend_recipes

model = build_network()

app = Flask(__name__)

VALID_CUISINES = {"any", "asian", "italian", "mexican", "american"}
VALID_DIETS = {"none", "vegetarian"}
VALID_COOK_TIMES = {"any", "short", "medium", "long"}

RECIPE_LOOKUP = {r["name"]: r for r in RECIPES}

RECIPE_DESCRIPTIONS = {
    "Chicken Stir Fry": "Tender chicken with crisp broccoli and garlic tossed in a savory soy sauce.",
    "Spaghetti Bolognese": "Classic Italian meat sauce slow-cooked with tomatoes and herbs over pasta.",
    "Veggie Buddha Bowl": "Wholesome quinoa bowl loaded with roasted veggies and tahini dressing.",
    "Beef Tacos": "Seasoned ground beef in warm tortillas with fresh lettuce, cheese, and salsa.",
    "Salmon Teriyaki": "Glazed salmon fillet with teriyaki sauce served over steamed rice and broccoli.",
    "Margherita Pizza (Homemade)": "Crispy homemade crust topped with fresh mozzarella, tomato sauce, and basil.",
    "Shrimp Alfredo": "Creamy parmesan pasta tossed with garlic butter shrimp.",
    "Black Bean Burritos": "Hearty black bean and rice burritos loaded with cheese, salsa, and sour cream.",
    "Grilled Chicken Salad": "Light grilled chicken over crisp greens with cucumber, tomato, and lemon.",
    "Pad Thai": "Classic Thai stir-fried rice noodles with shrimp, egg, peanuts, and lime.",
    "Egg Fried Rice": "Simple and satisfying fried rice with scrambled egg and soy sauce.",
    "Chicken Tacos": "Juicy seasoned chicken in warm tortillas with fresh salsa and cheese.",
    "Veggie Pasta": "Light pasta tossed with sautéed bell pepper, mushroom, and garlic in tomato sauce.",
    "Caesar Salad": "Classic crisp romaine with parmesan, croutons, and tangy caesar dressing.",
    "Chicken Noodle Soup": "Comforting homestyle chicken soup with vegetables and tender noodles.",
    "Shrimp Fried Rice": "Quick Asian-style fried rice packed with shrimp, egg, and green onion.",
    "Black Bean Soup": "Hearty Mexican-spiced black bean soup with garlic and tomato.",
    "Mushroom Risotto": "Creamy Italian risotto with sautéed mushrooms and aged parmesan.",
}

RECIPE_INSTRUCTIONS = {
    "Chicken Stir Fry": "1. Heat oil in a wok over high heat.\n2. Cook chicken until golden, about 5 minutes.\n3. Add garlic, broccoli, and soy sauce, stir fry 3–4 minutes.\n4. Serve over cooked rice.",
    "Spaghetti Bolognese": "1. Brown ground beef with diced onion and garlic.\n2. Add tomato sauce and simmer on low for 20 minutes.\n3. Cook pasta according to package directions.\n4. Serve sauce over pasta.",
    "Veggie Buddha Bowl": "1. Roast cubed sweet potato at 400°F for 25 minutes.\n2. Cook quinoa per package directions.\n3. Warm chickpeas in a pan.\n4. Assemble bowl and drizzle with tahini.",
    "Beef Tacos": "1. Brown ground beef with diced onion until cooked through.\n2. Warm tortillas in a dry pan.\n3. Fill with beef, lettuce, cheese, and salsa.\n4. Serve with lime wedges.",
    "Salmon Teriyaki": "1. Mix soy sauce, ginger, and sesame oil for marinade.\n2. Marinate salmon 15 minutes.\n3. Cook rice and steam broccoli.\n4. Sear salmon in a pan 4 minutes per side until glazed.",
    "Margherita Pizza (Homemade)": "1. Mix flour, yeast, and water into dough, rest 1 hour.\n2. Roll dough and spread tomato sauce.\n3. Top with mozzarella and basil.\n4. Bake at 475°F for 10–12 minutes until golden.",
    "Shrimp Alfredo": "1. Cook pasta and set aside.\n2. Melt butter, sauté garlic, then cook shrimp until pink.\n3. Add heavy cream and parmesan, simmer until thickened.\n4. Toss with pasta and serve.",
    "Black Bean Burritos": "1. Heat black beans and rice until warm.\n2. Warm tortillas in a pan.\n3. Fill with beans, rice, cheese, salsa, and sour cream.\n4. Roll up tightly and serve.",
    "Grilled Chicken Salad": "1. Season chicken and grill 5–6 minutes per side.\n2. Slice and rest 5 minutes.\n3. Toss lettuce, tomato, cucumber with olive oil and lemon.\n4. Top with chicken and serve.",
    "Pad Thai": "1. Soak rice noodles in warm water 20–30 minutes, drain.\n2. Stir fry shrimp until pink, set aside.\n3. Add noodles, egg, and sauce to pan, toss together.\n4. Top with peanuts, bean sprouts, and lime.",
    "Egg Fried Rice": "1. Use day-old cooked rice for best results.\n2. Scramble eggs in a hot oiled pan, set aside.\n3. Fry garlic and green onion, add rice and soy sauce.\n4. Mix in egg and serve hot.",
    "Chicken Tacos": "1. Season and cook chicken in a pan until done, then slice.\n2. Warm tortillas in a dry pan.\n3. Fill with chicken, salsa, cheese, and lettuce.\n4. Finish with a squeeze of lime.",
    "Veggie Pasta": "1. Cook pasta until al dente.\n2. Sauté garlic, bell pepper, and mushroom in olive oil.\n3. Add tomato sauce and simmer 10 minutes.\n4. Toss with pasta and serve.",
    "Caesar Salad": "1. Tear lettuce into bite-sized pieces.\n2. Mix olive oil, lemon, and garlic for dressing.\n3. Toss lettuce with dressing.\n4. Top with parmesan and croutons and serve.",
    "Chicken Noodle Soup": "1. Simmer chicken with garlic and onion in water for 20 minutes.\n2. Remove chicken, shred, and return to pot.\n3. Add carrot, celery, and noodles.\n4. Cook until noodles are tender, season and serve.",
    "Shrimp Fried Rice": "1. Use cold cooked rice for best texture.\n2. Stir fry shrimp until pink, set aside.\n3. Add rice, soy sauce, and green onion, stir fry on high heat.\n4. Mix in scrambled egg and shrimp, serve hot.",
    "Black Bean Soup": "1. Sauté diced onion and garlic in oil until soft.\n2. Add black beans, tomato, cumin, and broth.\n3. Simmer on medium heat for 20 minutes.\n4. Partially blend for a thicker texture and serve.",
    "Mushroom Risotto": "1. Sauté diced onion and garlic in butter until soft.\n2. Add rice and toast 2 minutes.\n3. Add broth one ladle at a time, stirring constantly.\n4. Fold in mushrooms and parmesan, serve immediately.",
}


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        error=None,
        budget="",
        pantry="",
        cuisine="any",
        diet="none",
        cook_time="any",
    )


@app.route("/results", methods=["POST"])
def results():
    budget_raw = request.form.get("budget", "").strip()
    pantry_raw = request.form.get("pantry", "").strip()
    cuisine = request.form.get("cuisine", "any").strip().lower()
    diet = request.form.get("diet", "none").strip().lower()
    cook_time = request.form.get("cook_time", "any").strip().lower()

    error = None

    if not budget_raw:
        error = "Please enter a budget."
    elif not pantry_raw:
        error = "Please enter at least one pantry ingredient."
    elif cuisine not in VALID_CUISINES:
        error = "Please choose a valid cuisine."
    elif diet not in VALID_DIETS:
        error = "Please choose a valid diet preference."
    elif cook_time not in VALID_COOK_TIMES:
        error = "Please choose a valid cook time."

    budget = None
    if error is None:
        try:
            budget = float(budget_raw)
            if budget <= 0:
                error = "Budget must be greater than 0."
        except ValueError:
            error = "Budget must be a valid number."

    if error:
        return render_template(
            "index.html",
            error=error,
            budget=budget_raw,
            pantry=pantry_raw,
            cuisine=cuisine,
            diet=diet,
            cook_time=cook_time,
        )

    pantry = [item.strip().lower() for item in pantry_raw.split(",") if item.strip()]

    try:
        top_recipes = recommend_recipes(
            model,
            pantry,
            budget,
            cuisine_pref=cuisine if cuisine != "any" else None,
            diet_pref=diet,
            cook_time_pref=cook_time if cook_time != "any" else None,
        )

        if cuisine != "any":
            top_recipes = [r for r in top_recipes if r["cuisine"] == cuisine]
        if cook_time != "any":
            top_recipes = [r for r in top_recipes if r["cook_time"] == cook_time]
        top_recipes = [r for r in top_recipes if r["ingredients_owned"] > 0]

        for r in top_recipes:
            name = r["recipe"]
            rec = RECIPE_LOOKUP.get(name, {})
            r["description"] = RECIPE_DESCRIPTIONS.get(name, "")
            r["instructions"] = RECIPE_INSTRUCTIONS.get(name, "")
            r["ingredients"] = rec.get("ingredients", [])

    except Exception as e:
        error = f"Recommendation failed: {str(e)}"
        return render_template(
            "index.html",
            error=error,
            budget=budget_raw,
            pantry=pantry_raw,
            cuisine=cuisine,
            diet=diet,
            cook_time=cook_time,
        )

    return render_template(
        "results.html",
        recipes=top_recipes,
        budget=f"{budget:.2f}",
        pantry=", ".join(pantry),
        cuisine=cuisine,
        diet=diet,
        cook_time=cook_time,
    )


if __name__ == "__main__":
    app.run(debug=True)
