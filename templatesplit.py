# Find subgroups based on dcol
def find_subgroups(dcol, df, gender):
    label = []
    subgroup = []
    for c in dcol:
        print(c)
        subgroup.append(df[df[c]==1])
        label.append((gender+"/"+c))
    
    return label, subgroup
    
def main():
    df = pd.read_csv("../clean-data/current_data.csv")

    # Create subgroups in based on following criteria
    # Split based on gender
    male_df = df[df['Gender_bin'] == 0]
    female_df = df[df['Gender_bin'] == 1]

    # Split based on Occupation
    d_col = ['Occupation_Corporate', 'Occupation_Housewife', 'Occupation_Others', 'Occupation_Student']
    lab0, arr0_df = find_subgroups(d_col, male_df, "Male")
    lab1, arr1_df = find_subgroups(d_col, female_df, "Female")

    print(lab0)
    print(arr0_df)
    print(lab1)
    print(arr1_df)