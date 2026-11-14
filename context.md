Act as a senior Python developer, cricket data scientist, and machine learning engineer.

I am participating in the Rajasthan Royals SupeRR Selector Season 2 Hackathon.

My goal is to build a high-quality, statistically reliable, explainable solution that identifies and ranks the top 10 batters and top 10 bowlers who perform best under pressure using the official provided cricket dataset.

PROJECT INPUT:
The official dataset is available in data_pack.zip, or already extracted into the data/ directory. Locate the files before starting. The pack contains the official Round 1 problem statement and CSV files such as ball_by_ball.csv, matches.csv, playing_eleven.csv, competitions.csv, teams.csv, and competition_team_player_mapping.csv.

IMPORTANT INSTRUCTIONS:
1. Read the complete official problem statement first. Identify the exact objectives, eligibility conditions, submission requirements, judging criteria and restrictions. Do not invent requirements.
2. Inspect every CSV file. Report row counts, column names, data types, missing values, duplicates, unique identifiers, competition coverage and date ranges.
3. Determine the correct relationships between matches, competitions, teams, players and ball-by-ball records.
4. Propose a statistically defensible definition of pressure based on match state before each ball, including wickets lost, required run rate, balls remaining, innings stage and match importance where supported.
5. Design separate pressure-performance models for batting and bowling. Include sample-size safeguards, consistency, context adjustment and explainable rankings.
6. Use only the official supplied dataset for the core rankings. Do not invent missing information or use unsupported external statistics.
7. Create a clean Python project structure with data/, src/, notebooks/, tests/, outputs/, reports/, requirements.txt, README.md and .gitignore.
8. Implement initial data loading, schema validation, cleaning and exploratory analysis using Pandas and NumPy.
9. Write tests for preprocessing and verify that joins do not multiply delivery records.
10. Prepare an implementation roadmap covering pressure scoring, batting analysis, bowling analysis, ranking validation, visualizations and final methodology documentation.

TECHNICAL STACK:
Python, Pandas, NumPy, SciPy, scikit-learn, Matplotlib, Jupyter Notebook and pytest.

DEVELOPMENT RULES:
- Work locally and keep the repository compatible with private GitHub version control.
- Do not create a website, frontend, chatbot or deployment configuration.
- Do not commit or publish raw hackathon datasets without checking competition permissions.
- Do not push code or create commits without my approval.
- Handle wides, no-balls, extras, dismissals, super overs and revised targets correctly.
- Prevent data leakage by computing pressure from the situation before each delivery.
- Explain assumptions and flag unsupported calculations.
- Run tests and report actual results, including failures.

START NOW:
First inspect the official problem statement and dataset. Then create the project structure, implement data loading and preprocessing, and run initial validation tests.

After completing this initial phase, show:
1. Official requirements summary.
2. Dataset exploration results.
3. Created files and their purposes.
4. Tests executed and results.
5. Data quality issues and assumptions.
6. Recommended pressure-model design.
7. Exact next implementation steps.

Do not proceed to the final player rankings until the dataset validation and pressure methodology are reviewed.