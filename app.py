import json
import io
from typing import Dict, List

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Yuk Cut Analysis", page_icon="🧱", layout="wide")

st.title("🧱 Let's Analyze Some Cuts!")
st.caption(
    "Add cuts to the dataset one row at a time using the dropdowns below.\n"
    "You can insert rows at a specific index if you missed a cut, \
    edit past rows if something needs to be changed, reorder them, or delete them. \n"
    "Please annotate all downfield cuts in order by when the cutter started cutting. \n"
    "If the disc is reset, simply mark 'Reset' for all columns. This applies to all uplines \
    and swings, as well as other handler passes. If someone cuts for a reset but does not receive \
    the disc, do not fill out a row for that cut."
)

# ---------- Defaults & Session State ----------

DEFAULT_SCHEMA = {
    # Feel free to change these to whatever columns & choices you want.
    "Point ID": 0,
    "Poss ID": 0,
    "Cut ID": 0,
    "Throw ID": 0,
    "Cutter": ["Abhi", "Ace (Sam D)", "Aidan", "Alex", "Christian", "Colin", \
               "Drizz (Brandon)", "Ethan", "Evan", "Henry", "Ivan", "James", \
                "Jay", "Jing", "Kyan", "Kyle", "Lex", "Luke", "Other", \
                "P (Matt P)", "Paul", "Rishabh", "Sam Y", "Simon", "Soumik", \
                "Spence", "Unknown", "Yang (Matt Y)", "Zach"],
    "Good Prework?": ["Yes", "No", "Unsure", "Reset"],
    "Good Timing?": ["Yes", "No", "Unsure", "Reset"],
    "Good Clear?": ["Yes", "No", "Unsure", "N/A", "Reset"],
    "Good Intensity?": ["Yes", "No", "Unsure", "Reset"],
    "Result": ["Reception", "Drop", "Clear", "Block", "Goal", "Throwaway"],
}

# index_columns = ["Point ID", "Poss ID", "Cut ID", "Throw ID"]

def ensure_state():
    if "schema" not in st.session_state:
        st.session_state.schema = DEFAULT_SCHEMA.copy()
    if "data" not in st.session_state:
        st.session_state.data = pd.DataFrame(columns=list(st.session_state.schema.keys()))
        # st.session_state.data = pd.DataFrame(columns=list(st.session_state.schema.keys()))
    if "next_id" not in st.session_state:
        # Optional internal identifier if you want to keep stable row IDs.
        st.session_state.next_id = 1


# list(schema.keys())

def init_empty_df(schema: Dict[str, List[str]]):
    # Create an empty dataframe with the schema's columns
    return pd.DataFrame(columns=list(schema.keys()))
    # return pd.DataFrame(list(schema.keys()))

ensure_state()


with st.sidebar:
    st.header("⚙️ Configure")
    
    st.subheader("📥 Upload to Resume Cutting Analysis")
    uploaded = st.file_uploader("Load an existing dataset and analyze further", type=["csv"])
    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            # st.session_state.data = coerce_df_to_schema(df_in, st.session_state.schema)
            st.session_state.data = df_in
            st.success(f"Loaded {len(st.session_state.data)} rows from CSV.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    st.subheader("📤 Download Cutting Dataset")

    # Film type
    download_type = st.selectbox("Film type", ["Game", "Practice"], key="download_type")

    # Opponent: plain text input with default 'Pitt C' (no dropdown)
    opponent = st.text_input("Opponent (for games)", value="Pitt C", key="opponent_text")

    # Film date
    film_date = st.date_input("Film date", key="film_date")

    # Start timestamp: plain text input with default 'HH:MM:SS'. Accept 'MM:SS' (treated as 0:MM:SS).
    start_text = st.text_input("Start timestamp (HH:MM:SS or MM:SS)", value="HH:MM:SS", key="start_text")

    def _sanitize(s: str) -> str:
        return "".join([c if c.isalnum() or c in ("-","_") else "_" for c in str(s)]).strip("_")

    def parse_time_input(s: str):
        s = (s or "").strip()
        if not s or s.upper() == "HH:MM:SS":
            return None
        parts = s.split(":")
        try:
            if len(parts) == 2:
                # MM:SS
                m = int(parts[0]); sec = int(parts[1]); h = 0
            elif len(parts) == 3:
                h = int(parts[0]); m = int(parts[1]); sec = int(parts[2])
            else:
                return None
            if not (0 <= m < 60 and 0 <= sec < 60 and h >= 0):
                return None
            return (h, m, sec)
        except Exception:
            return None

    parsed_time = parse_time_input(start_text)

    date_str = film_date.isoformat() if film_date is not None else "unknown-date"
    type_str = download_type.lower()

    # Opponent string for filename
    opp_str = _sanitize(opponent) if download_type == "Game" else "all"

    # Build time string in HHh-MMm-SSs format if parsed
    if parsed_time:
        h, m, sec = parsed_time
        time_str = f"{h:02d}h-{m:02d}m-{sec:02d}s"
    else:
        time_str = "HHh-MMm-SSs"

    filename = f"{date_str}_{time_str}_{type_str}_{(opp_str or 'unknown')}_dataset.csv"

    csv_bytes = st.session_state.data.to_csv(index=False).encode("utf-8")

    # Validation & warnings: require opponent change from default 'Pitt C' for games, and a valid timestamp
    download_disabled = False
    if download_type == "Game" and (not opponent or opponent.strip() == "" or opponent.strip() == "Pitt C"):
        download_disabled = True
        st.warning("Please enter the real opponent name before downloading a game file (default 'Pitt C' is not allowed).")
    if parsed_time is None:
        download_disabled = True
        st.warning("Please enter a valid start timestamp (HH:MM:SS or MM:SS).")

    st.download_button("Download current dataset", data=csv_bytes, file_name=filename, mime="text/csv", disabled=download_disabled)

    st.divider()
    if st.button("🗑️ Clear dataset", help="Remove all rows and start fresh."):
        st.session_state.data = init_empty_df(st.session_state.schema)
        st.success("Dataset cleared.")


def add_row_from_selection(at_index: int | None = None):
    """Create a row dict from current 'builder_' selections and insert/append it."""
    row = {}
    for c, choices in st.session_state.schema.items():
        val = st.session_state.get(f"builder_{c}")
        # If user hasn't picked, default to first option
        if val is None and choices:
            val = choices[0]
        row[c] = val

    df = st.session_state.data.copy()
    if len(df) == 0:

        if row["Result"] in ["Goal", "Block", "Drop", "Throwaway", "Reception"]:
            throw_id = 1
        else:
            throw_id = 0

        index_inputs = {"Point ID": 1, "Poss ID": 1, 
                        "Cut ID": 1, "Throw ID": throw_id}

        row.update(index_inputs)

        df.loc[len(df)] = row

    elif at_index is None and len(df) != 0:

        index_inputs = set_index_inputs(row, df)
        row.update(index_inputs)
        df.loc[len(df)] = row
    else:
        top = df.iloc[:at_index].copy()
        bottom = df.iloc[at_index:].copy()

        # Compute index values for the new row based on the top (previous rows)
        index_inputs = set_index_inputs(row, top)
        row.update(index_inputs)
        new_row_df = pd.DataFrame([row], columns=df.columns)

        # Recompute index columns for all rows that were below the insertion point
        # using the new row as the previous row for the bottom slice.
        adjusted_bottom = adjust_below_rows(new_row_df.iloc[0], bottom.copy())

        df = pd.concat([top, new_row_df, adjusted_bottom], ignore_index=True)

    st.session_state.data = df

def set_index_inputs(row, df):
    # df may be a DataFrame (slice of previous rows) or a Series representing the previous row.
    if isinstance(df, pd.Series):
        prev_row = df
    else:
        # assume DataFrame-like
        prev_row = df.iloc[len(df) - 1]

    # row may be a Series, dict, or similar mapping. Access with [] works for all.
    if prev_row["Result"] == "Goal":
        point_id = prev_row["Point ID"] + 1
        poss_id = 1
        cut_id = 1
        throw_id = 1 if row["Result"] != "Clear" else 0
    elif prev_row["Result"] in ["Block", "Drop", "Throwaway"]:
        point_id = prev_row["Point ID"]
        poss_id = prev_row["Poss ID"] + 1
        cut_id = 1
        throw_id = 1 if row["Result"] != "Clear" else 0
    else:
        point_id = prev_row["Point ID"]
        poss_id = prev_row["Poss ID"]
        cut_id = prev_row["Cut ID"] + 1
        throw_id = prev_row["Throw ID"] + 1 if row["Result"] != "Clear" else prev_row["Throw ID"]

    index_inputs = {"Point ID": point_id, "Poss ID": poss_id, 
                    "Cut ID": cut_id, "Throw ID": throw_id}

    return index_inputs

def adjust_below_rows(row, lower_data):
    # Reset index so integer positional labels 0..n-1 are available for .at lookups
    lower_data = lower_data.reset_index(drop=True)

    for i in range(len(lower_data)):
        lower_row = lower_data.iloc[i]
        index_inputs = set_index_inputs(lower_row, row)
        lower_data.at[i, "Point ID"] = index_inputs["Point ID"]
        lower_data.at[i, "Poss ID"] = index_inputs["Poss ID"]
        lower_data.at[i, "Cut ID"] = index_inputs["Cut ID"]
        lower_data.at[i, "Throw ID"] = index_inputs["Throw ID"]
        row = lower_data.iloc[i]
    return lower_data

# def set_point_id(row, prev_row):
#     if prev_row["Result"] == "Goal":
#         point_id = prev_row["Point ID"] + 1
#     else:
#         point_id = prev_row["Point ID"]
#     return point_id

# def set_poss_id(row, prev_row):
#     if prev_row["Result"] in ["Goal", "Block", "Drop", "Throwaway"]:
#         poss_id = prev_row["Poss ID"] + 1
#     else:
#         poss_id = prev_row["Poss ID"]
#     return poss_id

# def set_throw_id(row, prev_row):
#     if row["Result"] in ["Goal", "Block", "Drop", "Throwaway", "Reception"]:
#         throw_id = prev_row["Throw ID"] + 1
#     else:
#         throw_id = prev_row["Throw ID"]
#     return throw_id

# def set_index_vars(row, df):

#     prev_row = df.loc[len(df)-1]
#     point_id = set_point_id(row, prev_row)
#     poss_id = set_poss_id(row, prev_row)
#     throw_id = set_throw_id(row, prev_row)
#     cut_id = prev_row["Cut ID"]
#     index_inputs = {"Point ID": point_id, "Poss ID": poss_id, 
#                     "Cut ID": cut_id, "Throw ID": throw_id}

#     return index_inputs

def update_row(idx: int):
    df = st.session_state.data.copy()
    if not (0 <= idx < len(df)):
        st.warning("No such row to update.")
        return
    # Don't allow manual edits to index columns; only update editable columns
    index_cols = {"Point ID", "Poss ID", "Cut ID", "Throw ID"}
    for c in st.session_state.schema.keys():
        if c in index_cols:
            continue
        df.at[idx, c] = st.session_state.get(f"edit_{c}", df.at[idx, c])
    # Recompute index columns for the edited row based on the previous row (if any)
    if idx == 0:
        # First row: set initial indices
        if df.at[idx, "Result"] in ["Goal", "Block", "Drop", "Throwaway", "Reception"]:
            throw_id = 1
        else:
            throw_id = 0
        df.at[idx, "Point ID"] = 1
        df.at[idx, "Poss ID"] = 1
        df.at[idx, "Cut ID"] = 1
        df.at[idx, "Throw ID"] = throw_id
    else:
        # Use set_index_inputs with the slice of rows up to idx to compute based on the previous row
        index_inputs = set_index_inputs(df.iloc[idx].to_dict(), df.iloc[:idx].copy())
        df.at[idx, "Point ID"] = index_inputs["Point ID"]
        df.at[idx, "Poss ID"] = index_inputs["Poss ID"]
        df.at[idx, "Cut ID"] = index_inputs["Cut ID"]
        df.at[idx, "Throw ID"] = index_inputs["Throw ID"]

    # Adjust all rows below the edited row to ensure indices remain consistent
    if idx < len(df) - 1:
        lower = df.iloc[idx+1:].copy()
        adjusted_lower = adjust_below_rows(df.iloc[idx], lower)
        # Write back only the index columns from adjusted_lower into df
        for i in range(len(adjusted_lower)):
            for col in ["Point ID", "Poss ID", "Cut ID", "Throw ID"]:
                df.at[idx + 1 + i, col] = adjusted_lower.at[i, col]

    st.session_state.data = df

def delete_row(idx: int):
    df = st.session_state.data.copy()
    if 0 <= idx < len(df):
        df = df.drop(index=idx).reset_index(drop=True)
        st.session_state.data = df

def move_row(idx: int, direction: str):
    df = st.session_state.data.copy()
    if direction == "up" and idx > 0:
        # Swap rows
        df.iloc[idx-1], df.iloc[idx] = df.iloc[idx].copy(), df.iloc[idx-1].copy()
        df = df.reset_index(drop=True)
        # Recompute indices starting at the earlier swapped row
        start = idx-1
        if start == 0:
            prev = None
        else:
            prev = df.iloc[:start].iloc[-1]
        # Use set_index_inputs to recompute for the start row based on prev
        if prev is None:
            # First row: set initial indices
            if df.at[start, "Result"] in ["Goal", "Block", "Drop", "Throwaway", "Reception"]:
                throw_id = 1
            else:
                throw_id = 0
            df.at[start, "Point ID"] = 1
            df.at[start, "Poss ID"] = 1
            df.at[start, "Cut ID"] = 1
            df.at[start, "Throw ID"] = throw_id
        else:
            index_inputs = set_index_inputs(df.iloc[start].to_dict(), df.iloc[:start].copy())
            df.at[start, "Point ID"] = index_inputs["Point ID"]
            df.at[start, "Poss ID"] = index_inputs["Poss ID"]
            df.at[start, "Cut ID"] = index_inputs["Cut ID"]
            df.at[start, "Throw ID"] = index_inputs["Throw ID"]

        # Adjust all rows below start
        if start < len(df) - 1:
            lower = df.iloc[start+1:].copy()
            adjusted_lower = adjust_below_rows(df.iloc[start], lower)
            for i in range(len(adjusted_lower)):
                for col in ["Point ID", "Poss ID", "Cut ID", "Throw ID"]:
                    df.at[start + 1 + i, col] = adjusted_lower.at[i, col]

        st.session_state.data = df
    elif direction == "down" and idx < len(df) - 1:
        # Swap rows
        df.iloc[idx+1], df.iloc[idx] = df.iloc[idx].copy(), df.iloc[idx+1].copy()
        df = df.reset_index(drop=True)
        # Recompute indices starting at the earlier of the two swapped rows
        start = idx
        if start == 0:
            prev = None
        else:
            prev = df.iloc[:start].iloc[-1]

        if prev is None:
            if df.at[start, "Result"] in ["Goal", "Block", "Drop", "Throwaway", "Reception"]:
                throw_id = 1
            else:
                throw_id = 0
            df.at[start, "Point ID"] = 1
            df.at[start, "Poss ID"] = 1
            df.at[start, "Cut ID"] = 1
            df.at[start, "Throw ID"] = throw_id
        else:
            index_inputs = set_index_inputs(df.iloc[start].to_dict(), df.iloc[:start].copy())
            df.at[start, "Point ID"] = index_inputs["Point ID"]
            df.at[start, "Poss ID"] = index_inputs["Poss ID"]
            df.at[start, "Cut ID"] = index_inputs["Cut ID"]
            df.at[start, "Throw ID"] = index_inputs["Throw ID"]

        # Adjust all rows below start
        if start < len(df) - 1:
            lower = df.iloc[start+1:].copy()
            adjusted_lower = adjust_below_rows(df.iloc[start], lower)
            for i in range(len(adjusted_lower)):
                for col in ["Point ID", "Poss ID", "Cut ID", "Throw ID"]:
                    df.at[start + 1 + i, col] = adjusted_lower.at[i, col]

        st.session_state.data = df

def filter_dictionary(dct, remove_list):
    return {k : v for k, v in dct.items() if k not in remove_list}

# # ---------- Layout ----------
# left, right = st.columns([7, 5])


# ---------- Builder (left): add or insert using dropdowns ----------
st.subheader("➕ Build a row with dropdowns")

filtered_schema = filter_dictionary(st.session_state.schema, ["Point ID", "Poss ID", "Cut ID", "Throw ID"])
# Show column pickers (builder_)
pick_cols = st.columns(len(filtered_schema) or 1)
for i, (col_name, options) in enumerate(filtered_schema.items()):
    with pick_cols[i]:

        st.selectbox(
            col_name,
            options=options,
            key=f"builder_{col_name}",
            index=0 if options else None,
        )

c1, c2 = st.columns([1, 2])
with c1:
    if st.button("Add row to end", type="primary", use_container_width=True):
        add_row_from_selection(None)
        st.success("Row added to end.")
with c2:
    insert_at = st.number_input(
        "Insert at row index",
        min_value=0,
        max_value=max(len(st.session_state.data), 0),
        value=0,
        step=1,
        help="Choose the index to insert at (0 inserts at top).",
    )
    if st.button("Insert row at position", use_container_width=True):
        add_row_from_selection(int(insert_at))
        st.success(f"Row inserted at index {int(insert_at)}.")

st.divider()

st.subheader("📋 Current dataset")
if len(st.session_state.data) == 0:
    st.info("No rows yet. Use the controls above to add your first row.")
else:
    st.dataframe(
        st.session_state.data,
        use_container_width=True,
        hide_index=False,
    )

st.subheader("✏️ Edit / reorder / delete rows")

if len(st.session_state.data) == 0:
    st.caption("Rows will appear here once added.")
else:
    # Row selector as a dropdown of row indices with human-friendly labels
    def row_label(idx: int) -> str:
        try:
            row = st.session_state.data.iloc[idx]
            cutter = row.get("Cutter", "-")
            result = row.get("Result", "-")
            return f"{idx} — Cutter: {cutter}, Result: {result}"
        except Exception:
            return str(idx)

    row_count = len(st.session_state.data)
    row_options = list(range(row_count))
    # Use a selectbox for selecting the row; key holds the selected index
    selected_idx = st.selectbox(
        "Select row index to edit",
        options=row_options,
        index=0 if row_count > 0 else 0,
        key="edit_row_selector",
        format_func=row_label,
    )

    idx = int(st.session_state.edit_row_selector)

    # Prefill edit widgets with current row values
    current_row = st.session_state.data.iloc[int(idx)]
    # When selection changes, populate edit_ keys so widgets show current values
    if "last_edit_selection" not in st.session_state or st.session_state.last_edit_selection != idx:
        st.session_state.last_edit_selection = idx
        index_cols = {"Point ID", "Poss ID", "Cut ID", "Throw ID"}
        editable_items = [(col_name, options) for col_name, options in st.session_state.schema.items() if col_name not in index_cols]
        for col_name, options in editable_items:
            val = current_row[col_name]
            if options:
                try:
                    # If value exists in options, use it; otherwise default to first option
                    st.session_state[f"edit_{col_name}"] = val if val in options else options[0]
                except Exception:
                    st.session_state[f"edit_{col_name}"] = options[0]
            else:
                st.session_state[f"edit_{col_name}"] = val
    with st.container(border=True):
        st.markdown(f"**Editing row {int(idx)}**")
        edit_cols = st.columns(len(st.session_state.schema) or 1)
        # Hide index columns from the edit UI; only present editable columns
        index_cols = {"Point ID", "Poss ID", "Cut ID", "Throw ID"}
        editable_items = [(col_name, options) for col_name, options in st.session_state.schema.items() if col_name not in index_cols]
        edit_cols = st.columns(len(editable_items) or 1)
        for i, (col_name, options) in enumerate(editable_items):
            with edit_cols[i]:
                # Rely on st.session_state[f"edit_{col_name}"] to supply the current value
                if options:
                    st.selectbox(
                        f"{col_name}",
                        options=options,
                        key=f"edit_{col_name}",
                    )
                else:
                    st.text_input(f"{col_name}", key=f"edit_{col_name}")

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if st.button("Save changes", type="primary", use_container_width=True):
                update_row(int(idx))
                st.success(f"Row {int(idx)} updated.")
        with b2:
            if st.button("Delete row", use_container_width=True):
                delete_row(int(idx))
                st.success(f"Row {int(idx)} deleted.")
        with b3:
            if st.button("Move up", use_container_width=True, disabled=(idx == 0)):
                move_row(int(idx), "up")
        with b4:
            if st.button("Move down", use_container_width=True, disabled=(idx == len(st.session_state.data) - 1)):
                move_row(int(idx), "down")

st.divider()
st.subheader("🛠️ Tips")
st.markdown(
    "- Email Harty your CSV (ryan.harty24@gmail.com) or send over Slack when you are done!\n"
    "- **Insert row at position** lets you add a row between existing ones.\n"
    "- Use **Move up/Move down** to reorder.\n"
    "- If you want to take a break, download your dataset and re-upload it later to continue where you left off!\n"
    "- Let Harty know if you get any weird error messages or have suggestions for improvement.")