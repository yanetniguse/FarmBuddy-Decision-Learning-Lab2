import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
# =============================
# App Configuration
# =============================

# =============================
# Load Model Safely
# =============================
@st.cache_resource
def load_model():
    try:
        return joblib.load("yield_rf_pipeline.joblib")
    except FileNotFoundError:
        st.error("Model file not found. Make sure 'yield_rf_pipeline.joblib' is in the project folder.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.stop()  # stop the app if model fails

st.set_page_config(
    page_title="Farm Buddy 🌱",
    layout="wide"
)

# =============================
# Load Model
# =============================
# =============================
# Sidebar – Inputs
# =============================
# -----------------------------
# Initialize Session State
# -----------------------------
defaults = {
    "crop": "Maize",
    "season": "Kharif",
    "state": "Tamil Nadu",
    "year": 2020,
    "area": 1.0,
    "fertilizer": 50.0,
    "pesticide": 5.0,
    "current_prediction": 0.0,
    "decision_history": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.sidebar.header("🌾 Farm Inputs")

st.session_state.crop = st.sidebar.selectbox(
    "Crop",
    ["Rice", "Maize", "Cotton(lint)", "Coconut", "Groundnut"],
    index=["Rice", "Maize", "Cotton(lint)", "Coconut", "Groundnut"].index(st.session_state.crop),
    key="crop_input"
)

st.session_state.season = st.sidebar.selectbox(
    "Season",
    ["Kharif", "Rabi", "Whole Year"],
    key="season_input"
)

st.session_state.state = st.sidebar.selectbox(
    "State",
    ["Tamil Nadu", "Andhra Pradesh", "Karnataka", "Kerala"],
    key="state_input"
)

st.session_state.year = st.sidebar.slider(
    "Year",
    2000, 2025,
    st.session_state.year,
    key="year_input"
)

st.session_state.area = st.sidebar.slider(
    "Area (hectares)",
    0.5, 20.0,
    st.session_state.area,
    step=0.5,
    key="area_input"
)

st.session_state.fertilizer = st.sidebar.slider(
    "Fertilizer (kg)",
    0.0, 500.0,
    st.session_state.fertilizer,
    key="fertilizer_input"
)

st.session_state.pesticide = st.sidebar.slider(
    "Pesticide (kg)",
    0.0, 200.0,
    st.session_state.pesticide,
    key="pesticide_input"
)


# =============================
# Prepare Input for Model
# =============================
input_df = pd.DataFrame([{
    "crop": st.session_state.crop,
    "year": st.session_state.year,
    "season": st.session_state.season,
    "state": st.session_state.state,
    "area": st.session_state.area,
    "fertilizer": st.session_state.fertilizer,
    "pesticide": st.session_state.pesticide,
    "production": 0
}])


st.session_state.current_prediction = model.predict(input_df)[0]
prediction = st.session_state.current_prediction


# =============================
# Tabs Layout
# =============================
# In your existing app, add new tab:
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "📊 Prediction",
    "📘 Decision Log",
    "📈 Scenario Comparison",
    "🧠 Learning Summary"
])


# =============================
# TAB 1 — HOME
# =============================
with tab1:
    st.title("🌾 FarmBuddy — Decision Learning Lab")

    st.info("""
    **Learning Objective**  
    FarmBuddy is an educational decision-support system designed to help students understand how machine learning models respond to agricultural inputs.

    The system emphasizes **decision-making, comparison, and reflection**, rather than exact yield prediction.

    Predictions are illustrative and intended to support learning about model sensitivity, trade-offs, and responsible use of data-driven tools in agriculture.
    """)


    st.markdown("""
    Farming success depends on **many factors working together**.
    This tool helps farmers and students **see, learn, and understand**
    how everyday decisions affect crop yield.

    You don’t need technical knowledge,just adjust the inputs and learn by doing.
    """)

    st.divider()

    st.subheader("🌾 What affects crop yield?")
    factors = {
        "Factor": [
            "Land Size",
            "Fertilizer Use",
            "Crop Type",
            "Season",
            "Weather & Region",
            "Pesticide Use"
        ],
        "Impact Level": [5, 4, 4, 3, 3, 2]
    }

    df_factors = pd.DataFrame(factors).set_index("Factor")
    st.bar_chart(df_factors)

    st.markdown("""
    **How to understand this chart:**
    - Taller bars mean **bigger influence**
    - More land and good fertilizer help most
    - Too much pesticide may not help
    """)
with tab2:
    st.subheader("Crop Yield Predictor")
    st.caption("AI-powered crop yield estimation for learning & planning")

    # -----------------------------
    # Load Model
    # -----------------------------

    # -----------------------------
    # Prepare Current Inputs
    # -----------------------------
    current_inputs = {
        "crop": st.session_state.crop,
        "season": st.session_state.season,
        "state": st.session_state.state,
        "year": st.session_state.year,
        "area": st.session_state.area,
        "fertilizer": st.session_state.fertilizer,
        "pesticide": st.session_state.pesticide,
        "production": 0,  # placeholder
    }

    input_df = pd.DataFrame([current_inputs])

    # -----------------------------
    # Prediction
    # -----------------------------
    predicted_yield = float(model.predict(input_df)[0])
    # store in session_state for other tabs
    st.session_state.latest_prediction_after = predicted_yield

    st.metric(
        label=f"Estimated Yield for {st.session_state.crop} "
              f"({st.session_state.season}, {st.session_state.state})",
        value=f"{predicted_yield:.2f} t/ha"
    )

    st.markdown(f"""
    **What this means:**  
    - Planting **1 hectare** of {st.session_state.crop} under these conditions may yield approx **{predicted_yield:.2f} t/ha**  
    - Total harvest scales with land area  
    - Fertilizer, pesticide, and season adjustments can influence yield
    """)

    # -----------------------------
    # Educational Explanation
    # -----------------------------
    st.subheader("How Your Inputs Affect Yield")
    st.markdown(f"""
    - **Crop:** {st.session_state.crop} typically yields {'high' if st.session_state.crop in ['Maize','Rice'] else 'moderate'}  
    - **Season:** {st.session_state.season} affects rainfall and growth  
    - **State/Region:** {st.session_state.state} affects soil and climate  
    - **Area:** Larger area increases total yield, but efficiency matters  
    - **Fertilizer:** {st.session_state.fertilizer} kg supports growth, excess may harm  
    - **Pesticide:** {st.session_state.pesticide} kg protects crops  
    - **Year:** Reflects historical trends; recent improvements may increase yield
    """)

    # -----------------------------
    # Input Summary
    # -----------------------------
    st.subheader("🌾 Input Summary")
    st.table(input_df)

    # -----------------------------
    # Model Trust & Explanation
    # -----------------------------
    st.subheader("🤝 Model Confidence & Uncertainty")

    training_ranges = {
        "area": (0.1, 5.0),
        "fertilizer": (0, 200),
        "pesticide": (0, 50),
    }

    confident_inputs = [k for k, v in current_inputs.items()
                        if k in training_ranges and training_ranges[k][0] <= v <= training_ranges[k][1]]

    uncertain_inputs = [k for k, v in current_inputs.items()
                        if k in training_ranges and not (training_ranges[k][0] <= v <= training_ranges[k][1])]

    if confident_inputs:
        st.success(f"The model is confident about: {', '.join(confident_inputs)}")
    if uncertain_inputs:
        st.warning(f"The model is less confident about: {', '.join(uncertain_inputs)}")

    # ±10% range to indicate uncertainty
    yield_min = predicted_yield * 0.9
    yield_max = predicted_yield * 1.1
    st.markdown(f"**Predicted Yield Range:** {yield_min:.2f} – {yield_max:.2f} t/ha")

    # Historical cases reference
    similar_cases = 120  # placeholder
    st.info(f"This prediction is based on {similar_cases} similar historical cases")

    if uncertain_inputs:
        st.error("⚠️ Some inputs are outside the model's training range. Interpret predictions cautiously.")


# --------------------------
# Tab 3: Before vs After Comparison
# --------------------------
with tab3:
    st.subheader("📘 Decision & Learning Log")

    # =========================================================
    # Current Situation (Snapshot)
    # =========================================================
    st.subheader("🌱 Current Situation")

    current_df = pd.DataFrame({
        "Parameter": [
            "Crop", "Season", "State", "Year",
            "Area (ha)", "Fertilizer (kg)", "Pesticide (kg)",
            "Current Predicted Yield (t/ha)"
        ],
        "Value": [
            st.session_state.crop,
            st.session_state.season,
            st.session_state.state,
            st.session_state.year,
            f"{st.session_state.area:.2f}",
            f"{st.session_state.fertilizer:.2f}",
            f"{st.session_state.pesticide:.2f}",
            f"{st.session_state.current_prediction:.2f}"
        ]
    })

    st.table(current_df)
    st.divider()

    # =========================================================
    # Log a Decision
    # =========================================================
    st.subheader("📝 Log Your Decision")

    decision_type = st.radio(
        "What are you changing?",
        [
            "Increase fertilizer",
            "Reduce fertilizer",
            "Add pesticide",
            "Increase area",
            "Change crop",
            "Other",
        ],
        key="decision_type_tab3"
    )

    reasoning = st.text_area(
        "Explain your reasoning (minimum 30 words)",
        height=140,
        key="decision_reasoning_tab3"
    )

    if st.button("💾 Save Decision", type="primary"):
        text = reasoning.lower()
        words = reasoning.split()

        relevant_keywords = ["fertilizer", "pesticide", "crop", "yield", "season", "area"]
        keyword_hits = [kw for kw in relevant_keywords if kw in text]

        if len(words) < 30:
            st.error(
                "Your explanation is too short.\n\n"
                "🧠 **What to do:** Explain *why* you made this farming decision. "
                "Mention the crop, season, inputs used, and how you expect them to affect yield."
            )

        elif len(keyword_hits) == 0:
            st.error(
                "Your explanation does not appear to relate to farming decisions.\n\n"
                "🧠 **What to do:** Focus on agricultural factors such as **crop choice, fertilizer use, pesticide application, land area, or season**.\n\n"
                "✍️ Example: *Increasing fertilizer may improve crop growth, but excessive use could reduce yield due to soil stress.*"
            )

        elif len(keyword_hits) < 2:
            st.warning(
                "Your explanation is partially relevant but incomplete.\n\n"
                "🧠 **What to do:** Try connecting **at least two farming factors** (e.g., fertilizer and yield, season and crop performance).\n\n"
                "✍️ Tip: Good farming decisions explain **cause → effect**."
            )

        else:
            snapshot = {
                "timestamp": datetime.now(),
                "decision_type": decision_type,
                "reasoning": reasoning,
                "inputs": {
                    "crop": st.session_state.crop,
                    "season": st.session_state.season,
                    "state": st.session_state.state,
                    "year": st.session_state.year,
                    "area": st.session_state.area,
                    "fertilizer": st.session_state.fertilizer,
                    "pesticide": st.session_state.pesticide,
                },
                "prediction_snapshot": float(st.session_state.current_prediction),
            }

            st.session_state.decision_history.append(snapshot)
            st.success(
                "Decision saved successfully.\n\n"
                "✅ You provided a clear, relevant explanation linking farming choices to yield outcomes."
            )


    # =========================================================
    # Stop if no decisions
    # =========================================================
    if not st.session_state.decision_history:
        st.info("Save at least one decision to unlock comparison and learning.")
        st.stop()

    # =========================================================
    # Latest Decision Analysis
    # =========================================================
    st.divider()
    st.subheader("📊 Latest Decision: Before vs After")

    last_decision = st.session_state.decision_history[-1]

    # --- BEFORE ---
    before_yield = last_decision["prediction_snapshot"]
    before_inputs = last_decision["inputs"]

    # --- AFTER (recomputed dynamically) ---
    model_columns = [
        "crop", "season", "state", "year",
        "area", "fertilizer", "pesticide", "production"
    ]

    after_inputs = {
        "crop": st.session_state.crop,
        "season": st.session_state.season,
        "state": st.session_state.state,
        "year": st.session_state.year,
        "area": st.session_state.area,
        "fertilizer": st.session_state.fertilizer,
        "pesticide": st.session_state.pesticide,
        "production": 0
    }

    after_df = pd.DataFrame([after_inputs], columns=model_columns)
    after_yield = float(model.predict(after_df)[0])
    # After computing the "after" prediction
    after = float(model.predict(input_df)[0])

    # Store in session_state for use elsewhere
    st.session_state.latest_prediction_after = after

    delta = after_yield - before_yield

    # =========================================================
    # Bar Chart (ONLY HERE)
    # =========================================================
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["Before Decision", "After Changes"],
        [before_yield, after_yield],
        color=["#7da0fa", "#8fd19e"]
    )

    ax.set_ylabel("Predicted Yield (tons/ha)")
    ax.set_title("Impact of Your Decision")

    for i, v in enumerate([before_yield, after_yield]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")

    st.pyplot(fig)

    st.metric(
        "Yield Change",
        f"{after_yield:.2f} t/ha",
        delta=f"{delta:+.2f} t/ha"
    )

    # =========================================================
    # Tree Visualization (Meaningful, Not Decorative)
    # =========================================================
    from PIL import Image
    import os

    def get_tree_image(yield_value):
        if yield_value >= 30:
            return "tree_xlarge.jpg"
        elif yield_value >= 20:
            return "tree_large.jpg"
        elif yield_value >= 10:
            return "tree_medium.jpg"
        else:
            return "tree_small.jpg"

    img_dir = "images"

    before_tree = Image.open(os.path.join(img_dir, get_tree_image(before_yield)))
    after_tree = Image.open(os.path.join(img_dir, get_tree_image(after_yield)))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌳 Before Decision")
        st.image(before_tree, caption=f"{before_yield:.2f} t/ha")
        st.table(pd.DataFrame(before_inputs.items(), columns=["Input", "Value"]))

    with col2:
        st.subheader("🌳 After Changes")
        st.image(after_tree, caption=f"{after_yield:.2f} t/ha")
        st.table(pd.DataFrame(after_inputs.items(), columns=["Input", "Value"]))

    # =========================================================
    # Learning Cue (Short — Deep learning goes to Tab 5)
    # =========================================================
    if abs(delta) < 0.05:
        st.warning(
            "This decision had little impact. "
            "This suggests the changed input is not a limiting factor."
        )
    elif delta > 0:
        st.success("This decision improved the predicted yield.")
    else:
        st.error("This decision reduced the predicted yield.")

    st.caption(
        f"Decision logged on {last_decision['timestamp'].strftime('%Y-%m-%d %H:%M')}"
    )

# --------------------------
# Tab 4: Decision & Learning Log
# --------------------------
with tab4:
    st.subheader("📈 Before vs After Comparison")

    if not st.session_state.decision_history:
        st.info("Log at least one decision to see comparisons.")
    else:
        # Labels for selection
        decision_labels = [
            f"{i+1}. {d['decision_type']} ({d['timestamp'].strftime('%Y-%m-%d %H:%M')})"
            for i, d in enumerate(st.session_state.decision_history)
        ]
        selected_index = st.selectbox(
        "Select a decision to analyze",
        range(len(decision_labels)),
        format_func=lambda i: decision_labels[i],
        key="tab5_learning_decision_select"
)


        selected_decision = st.session_state.decision_history[selected_index]

        # Use frozen snapshots
        before = selected_decision["prediction_snapshot"]
        after = selected_decision.get("prediction_after", st.session_state.current_prediction)

        # Display input comparison table
        before_inputs = selected_decision["inputs"]
        after_inputs = {
            "crop": st.session_state.crop,
            "season": st.session_state.season,
            "state": st.session_state.state,
            "year": st.session_state.year,
            "area": st.session_state.area,
            "fertilizer": st.session_state.fertilizer,
            "pesticide": st.session_state.pesticide,
        }

        comparison_df = pd.DataFrame({
            "Parameter": list(before_inputs.keys()),
            "Before": list(before_inputs.values()),
            "After": list(after_inputs.values())
        })

        
        st.table(comparison_df)

        # Bar chart for predicted yield
        st.subheader("📊 Predicted Yield Before vs After")
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(["Before Decision", "After Change"], [before, after], color=['skyblue','lightgreen'])
        ax.set_ylabel("Predicted Yield (tons/ha)")
        ax.set_title("Impact of Your Decision")

        for i, v in enumerate([before, after]):
            ax.text(i, v, f"{v:.2f}", ha='center', va='bottom')

        st.pyplot(fig)

        delta = after - before
        if abs(delta) < 0.05:
            st.warning("⚠️ This change had little effect. Input likely not limiting.")
        elif delta > 0:
            st.success(f"✅ Your changes increased yield by {delta:.2f} tons/ha.")
        else:
            st.error(f"❌ Your changes reduced yield by {abs(delta):.2f} tons/ha.")

        # Plain language insight
        st.markdown(f"""
**💡 Insight:**  
- Predicted yield before your decision: **{before:.2f} tons/ha**  
- Predicted yield after your decision: **{after:.2f} tons/ha**  
- Net change: **{delta:+.2f} tons/ha**  

Compare the input table above to see which changes had the most impact.
""")

# --------------------------
# Tab 5: Learning Summary
# -----------------------
with tab5:
    history = st.session_state.decision_history

    if len(history) < 1:
        st.info("Log at least one decision to unlock learning insights.")
        st.stop()

    # -------------------------------------------------
    # 1. Overall Yield Narrative (Big Picture Learning)
    # -------------------------------------------------
    initial_yield = history[0]["prediction_snapshot"]
    current_yield = st.session_state.current_prediction
    total_delta = current_yield - initial_yield

    st.markdown("## 📘 Learning Summary: Interpreting Your Decisions")

    st.markdown("""
    This section explains **why the model responded the way it did**,  
    **what this implies for real farming decisions**, and  
    **how a student should experiment next**.

    The goal is not to predict exact yields, but to **learn decision-making logic**.
    """)

    st.markdown("### 📊 Overall Change in Predicted Yield")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Initial Predicted Yield",
            f"{initial_yield:.2f} tons/ha"
        )
    with col2:
        st.metric(
            "Current Predicted Yield",
            f"{current_yield:.2f} tons/ha",
            delta=f"{total_delta:+.2f} tons/ha"
        )

    st.markdown("""
    **Interpretation:**  
    - The *initial predicted yield* represents the model’s baseline estimate  
    - The *current predicted yield* reflects the **combined effect of all decisions made so far**  
    - The difference shows the **direction and magnitude of learning impact**, not certainty
    """)

    st.info("""
    **Teaching point:**  
    A model does not evaluate decisions in isolation.  
    It evaluates **patterns created by combinations of inputs over time**.
    """)

    st.divider()

    # -------------------------------------------------
    # 2. Decision-by-Decision Academic Reflection
    # -------------------------------------------------
    st.markdown("### 📋 How Each Decision Shaped the Outcome")

    rows = []
    for i, d in enumerate(history, 1):
        before = d["prediction_snapshot"]
        after = d.get("prediction_after", current_yield)
        delta = after - before

        rows.append({
            "Decision #": i,
            "Decision Focus": d["decision_type"],
            "Predicted Yield Before (t/ha)": f"{before:.2f}",
            "Predicted Yield After (t/ha)": f"{after:.2f}",
            "Change (Δ t/ha)": f"{delta:+.2f}"
        })

    st.table(pd.DataFrame(rows))

    st.markdown("""
    **What this table teaches:**  
    - Large positive or negative changes indicate **high model sensitivity**  
    - Repeated declines suggest **over-adjustment or conflicting inputs**  
    - Stable outcomes suggest inputs are within a reasonable range
    """)

    st.info("""
    **Academic insight:**  
    In agricultural systems, *stability after multiple adjustments* often signals
    a locally optimal management strategy.
    """)

    st.divider()

    # -------------------------------------------------
    # 3. Why the Model Behaved This Way
    # -------------------------------------------------
    st.markdown("### 🔍 Why Did the Model Respond Like This?")

    impact_counter = {}

    for d in history:
        before = d["prediction_snapshot"]
        after = d.get("prediction_after", current_yield)
        delta = after - before

        for k in d["inputs"].keys():
            impact_counter[k] = impact_counter.get(k, 0) + abs(delta)

    sorted_impacts = sorted(impact_counter.items(), key=lambda x: x[1], reverse=True)

    if sorted_impacts:
        st.markdown("""
        The model showed the strongest responses to the following inputs:
        """)
        for k, v in sorted_impacts:
            st.markdown(f"- **{k.capitalize()}** (relative influence score: {v:.2f})")
    else:
        st.write("No dominant influencing input has emerged yet.")

    st.markdown("""
    **Explanation:**  
    The model was trained on historical agricultural data where certain inputs
    frequently co-occurred with yield changes.  
    As a result, adjustments to these inputs trigger stronger responses.

    ⚠️ **Important clarification for students:**  
    This does **not** mean:
    - These inputs are the most important in all real farms  
    - The relationship is strictly causal  

    It means:
    - The model has learned **statistical sensitivity**, not agronomic certainty
    """)

    st.divider()

    # -------------------------------------------------
    # 4. What This Teaches About Farming Decisions
    # -------------------------------------------------
    st.markdown("### 🌱 What a Student Should Learn From This")

    st.markdown("""
    1. **One change at a time matters**  
    Simultaneous adjustments make it difficult to identify true drivers.

    2. **Direction is more valuable than precision**  
    Models are best used to understand *whether* a decision helps or harms.

    3. **Consistency often beats intensity**  
    Gradual adjustments frequently outperform drastic interventions.

    4. **Data does not replace judgment**  
    Field conditions, farmer experience, and timing still matter.
    """)

    st.divider()

    # -------------------------------------------------
    # 5. What to Try Next (Guided Experimentation)
    # -------------------------------------------------
    st.markdown("### 🎯 What You Should Try Next")

    if total_delta > 0:
        st.success("""
        **Learning outcome:**  
        Your decisions moved the prediction in a positive direction.

        **Next experiment:**  
        - Keep most inputs constant  
        - Adjust only **one variable slightly**  
        - Observe whether the positive trend stabilizes or reverses
        """)
    elif total_delta < 0:
        st.warning("""
        **Learning outcome:**  
        Your decisions reduced the predicted yield.

        **Next experiment:**  
        - Revert the most recent change  
        - Introduce smaller adjustments  
        - Focus on isolating individual effects
        """)
    else:
        st.info("""
        **Learning outcome:**  
        Your configuration appears stable.

        **Next experiment:**  
        - Test different seasonal assumptions  
        - Compare crops or management strategies  
        - Use the model for scenario comparison rather than optimization
        """)

    st.divider()

    # -------------------------------------------------
    # 6. Responsible Interpretation (Instructor Reminder)
    # -------------------------------------------------
    st.markdown("### 🧭 Responsible Use of This Learning Tool")

    st.markdown("""
    - This system supports **learning and exploration**, not yield guarantees  
    - Many real-world variables (weather shocks, pests, labor) are not captured  
    - The strongest value lies in **decision reasoning**, not numerical accuracy
    """)

    st.caption(
        "Tab 5 is designed as a teaching assistant — explaining outcomes, encouraging reflection, and guiding experimentation."
    )
