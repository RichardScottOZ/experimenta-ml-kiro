# IDEATION.md

## Product idea

Build a simple agentic experimentation engine for practical machine learning, driven by Kiro CLI agents.

The user provides:
- a mission
- a dataset
- constraints
- a budget

The system then:
- reasons about candidate recipes
- experiments automatically
- analyzes the results
- returns the best recipe, the corresponding pipeline, and a final report

This product is designed for ML settings where most gains do not come only from model tuning, but from:
- cleaning
- realistic splits
- transformations
- aggregations
- feature logic
- representation choices

---

## Mission

The mission of the product is to help users solve a concrete ML problem through controlled recipe search.

For each run, the system should understand:
- what problem is being solved
- what metric matters most
- what constraints must not be broken
- what segments matter
- how much search budget is available

---

## Why this is needed

Many ML workflows get stuck because experimentation is scattered and manual.

This product solves that by making the **recipe** the central artifact.

---

## Main bet

> In practical ML, autonomous search should happen over recipes, not only over models.

A recipe includes:
- data policy
- split policy
- feature blocks
- model configuration

---

## User promise

A user should be able to say:

> Here is my problem and my data.
> Try meaningful recipe variations for me.
> Show me what improved, what failed, and what I should keep.

---

## Output philosophy

The system should always produce three useful things:

### Recipe
A structured representation of the winning approach.

### Pipeline
A reproducible pipeline definition implementing that recipe.

### Report
A readable explanation of results, improvements, insights, and recommendations.
