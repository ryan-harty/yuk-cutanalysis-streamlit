import json
import io
from typing import Dict, List

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Row Builder", page_icon="🧱", layout="wide")

st.title("🧱 Let's Analyze Some Cuts!")
st.caption(
    "Add cuts to the dataset one at a time using the dropdowns below."
    "You can also insert rows between others, edit past rows, move them, or delete them."
)

# ---------- Defaults & Session State ----------

DEFAULT_SCHEMA = {
    # Feel free to change these to whatever columns & choices you want.
    "Point ID": 0,
    "Poss ID": 0,
    "Cut ID": 0,
    "Throw ID": 0,
    "Cutter": ["James", "Lex", "Abhi", "Lucas"],
    "Good Prework?": ["Yes", "No", "Unsure"],
    "Good Timing?": ["Yes", "No", "Unsure"],
    "Good Clear?": ["Yes", "No", "Unsure", "N/A"],
    "Good Intensity?": ["Yes", "No", "Unsure"],
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

def init_empty_df(schema: Dict[str, List[str]], index_columns) -> pd.DataFrame:
    return pd.DataFrame(list(schema.keys()))
    # return pd.DataFrame(list(schema.keys()))

ensure_state()


with st.sidebar:
    st.header("⚙️ Configure")
    
    st.subheader("📥 Upload CSV")
    uploaded = st.file_uploader("Load an existing dataset (must match schema columns)", type=["csv"])
    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            # st.session_state.data = coerce_df_to_schema(df_in, st.session_state.schema)
            st.session_state_data = df_in
            st.success(f"Loaded {len(st.session_state.data)} rows from CSV.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    st.subheader("📤 Download CSV")
    csv_bytes = st.session_state.data.to_csv(index=False).encode("utf-8")
    st.download_button("Download current dataset", data=csv_bytes, file_name="dataset.csv", mime="text/csv")

    st.divider()
    if st.button("🗑️ Clear dataset", help="Remove all rows and start fresh."):
        st.session_state.data = init_empty_df(index_columns, st.session_state.schema)
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

        index_inputs = set_index_inputs(row, top)
        row.update(index_inputs)
        new_row_df = pd.DataFrame([row], columns=df.columns)
        df = pd.concat([top, new_row_df, bottom], ignore_index=True)

    st.session_state.data = df

def set_index_inputs(row, df):

    prev_row = df.iloc[len(df)-1]
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
    for c in st.session_state.schema.keys():
        df.at[idx, c] = st.session_state.get(f"edit_{c}", df.at[idx, c])
    st.session_state.data = df

def delete_row(idx: int):
    df = st.session_state.data.copy()
    if 0 <= idx < len(df):
        df = df.drop(index=idx).reset_index(drop=True)
        st.session_state.data = df

def move_row(idx: int, direction: str):
    df = st.session_state.data.copy()
    if direction == "up" and idx > 0:
        df.iloc[idx-1], df.iloc[idx] = df.iloc[idx].copy(), df.iloc[idx-1].copy()
        st.session_state.data = df.reset_index(drop=True)
    elif direction == "down" and idx < len(df) - 1:
        df.iloc[idx+1], df.iloc[idx] = df.iloc[idx].copy(), df.iloc[idx+1].copy()
        st.session_state.data = df.reset_index(drop=True)

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
        "Insert above row index",
        min_value=0,
        max_value=max(len(st.session_state.data), 0),
        value=0,
        step=1,
        help="Choose the index to insert above (0 inserts at top).",
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
    idx = st.number_input(
        "Select row index to edit",
        min_value=0,
        max_value=len(st.session_state.data) - 1,
        value=0,
        step=1,
    )

    # Prefill edit widgets with current row values
    current_row = st.session_state.data.iloc[int(idx)]
    with st.container(border=True):
        st.markdown(f"**Editing row {int(idx)}**")
        edit_cols = st.columns(len(st.session_state.schema) or 1)
        for i, (col_name, options) in enumerate(st.session_state.schema.items()):
            default_val = current_row[col_name] if pd.notna(current_row[col_name]) else (options[0] if options else None)
            with edit_cols[i]:
                st.selectbox(
                    f"{col_name}",
                    options=options,
                    key=f"edit_{col_name}",
                    index=(options.index(default_val) if (default_val in options) else 0) if options else None,
                )

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
    "- Use **Edit schema** in the sidebar to change columns and dropdown choices.\n"
    "- **Insert row at position** lets you add a row between existing ones.\n"
    "- Use **Move up/Move down** to reorder.\n"
    "- **Upload CSV** will align columns to the current schema; unknown values become blank so you can fix them.")