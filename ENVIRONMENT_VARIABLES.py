TEAMS = ["Broncos", "Roosters", "Wests Tigers", "Rabbitohs", "Storm", "Eels", "Raiders", "Knights", "Dragons", "Sea Eagles", "Panthers", "Sharks", "Bulldogs", "Dolphins", "Titans", "Cowboys", "Warriors"]
NRL_WEBSITE = "https://www.nrl.com/draw/nrl-premiership/"

PLAYER_LABELS =     ["Number", "Position", "Mins Played", "Points", "Tries", "Conversions", "Conversion Attempts",
                    "Penalty Goals", "Goal Conversion Rate", "1 Point Field Goals",
                    "2 Point Field Goals", "Total Points", "All Runs", "All Run Metres",
                    "Kick Return Metres", "Post Contact Metres", "Line Breaks",
                    "Line Break Assists", "Try Assists", "Line Engaged Runs", "Tackle Breaks",
                    "Hit Ups", "Play The Ball", "Average Play The Ball Speed",
                    "Dummy Half Runs", "Dummy Half Run Metres", "One on One Steal",
                    "Offloads", "Dummy Passes", "Passes", "Receipts", "Passes To Run Ratio",
                    "Tackle Efficiency", "Tackles Made", "Missed Tackles",
                    "Ineffective Tackles", "Intercepts", "Kicks Defused", "Kicks",
                    "Kicking Metres", "Forced Drop Outs", "Bomb Kicks", "Grubbers",
                    "40/20", "20/40", "Cross Field Kicks", "Kicked Dead", "Errors",
                    "Handling Errors", "One on One Lost", "Penalties", "Ruck Infringements",
                    "Inside 10 Metres", "On Report", "Sin Bins", "Send Offs",
                    "Stint One", "Stint Two"]

PLAYER_STATS =     ["Mins Played", "Points", "Tries", "Conversions", "Conversion Attempts",
                    "Penalty Goals", "Goal Conversion Rate", "1 Point Field Goals",
                    "2 Point Field Goals", "Fantasy", "All Runs", "All Run Metres",
                    "Kick Return Metres", "Post Contact Metres", "Line Breaks",
                    "Line Break Assists", "Try Assists", "Line Engaged Runs", "Tackle Breaks",
                    "Hit Ups", "Play The Ball", "Average Play The Ball Speed",
                    "Dummy Half Runs", "Dummy Half Run Metres", "One on One Steal",
                    "Offloads", "Dummy Passes", "Passes", "Receipts", "Passes To Run Ratio",
                    "Tackle Efficiency", "Tackles Made", "Missed Tackles",
                    "Ineffective Tackles", "Intercepts", "Kicks Defused", "Kicks",
                    "Kicking Metres", "Forced Drop Outs", "Bomb Kicks", "Grubbers",
                    "40/20", "20/40", "Cross Field Kicks", "Kicked Dead", "Errors",
                    "Handling Errors", "One on One Lost", "Penalties", "Ruck Infringements",
                    "Inside 10 Metres", "On Report", "Sin Bins", "Send Offs",
                    "Stint One", "Stint Two"]

TEAM_STATS =     ["Points", "Tries", "Conversions", "Conversion Attempts",
                    "Penalty Goals", "1 Point Field Goals",
                    "2 Point Field Goals", "All Runs", "All Run Metres",
                    "Kick Return Metres", "Post Contact Metres", "Line Breaks",
                    "Line Break Assists", "Try Assists", "Line Engaged Runs", "Tackle Breaks",
                    "Hit Ups", 
                    "Dummy Half Runs", "Dummy Half Run Metres", "One on One Steal",
                    "Offloads", "Dummy Passes", "Passes", "Receipts", 
                     "Tackles Made", "Missed Tackles",
                    "Ineffective Tackles", "Intercepts", "Kicks Defused", "Kicks",
                    "Kicking Metres", "Forced Drop Outs", "Bomb Kicks", "Grubbers",
                    "40/20", "20/40", "Cross Field Kicks", "Kicked Dead", "Errors",
                    "Handling Errors", "One on One Lost", "Penalties", "Ruck Infringements",
                    "Inside 10 Metres", "On Report", "Sin Bins", "Send Offs"]

NRL_2024_ROUND = 1

ROUNDS = ['8']

TEAM_COLOURS = {
    "Broncos": "#760135",
    "Roosters": "#e82c2e",
    "Wests Tigers": "#f68b1f",
    "Rabbitohs": "#006633",
    "Storm": "#3E2783",
    "Eels": "#ffd327",
    "Raiders": "#c3d941",
    "Knights": "#ee3524",
    "Dragons": "#e2231b",
    "Sea Eagles": "#6F0F3B",
    "Panthers": "#221F20",
    "Sharks": "#00a9d8",
    "Bulldogs": "#0054A4",
    "Dolphins": "#E5CC7A", 
    "Titans": "#e7a614",
    "Cowboys": "#002b5c",
    "Warriors": "#231f20"
}


TEAM_COLOURS_INVERSE = {
    "Broncos": "#fbbf15",
    "Roosters": "#00305e",
    "Wests Tigers": "#000000",
    "Rabbitohs": "#e2261b",
    "Storm": "#f9b018",
    "Eels": "#006eb5",
    "Raiders": "#00ac5b",
    "Knights": "#00539f",
    "Dragons": "#ffffff",
    "Sea Eagles": "#ffffff",
    "Panthers": "#ff0082",
    "Sharks": "#000000",
    "Bulldogs": "#A7A9AC",
    "Dolphins": "#FB141E", 
    "Titans": "#009ddc",
    "Cowboys": "#ffdd00",
    "Warriors": "#bdbcbc"
}

types_to_convert = {
    
    # Strings
    'Player': 'string',
    'Position': 'categpry',
    'Number': 'categpry',
    'Team': 'string',

    
    # Integers
    'Points': 'int',
    'Tries': 'int',
    'Conversions': 'int',
    'Conversion Attempts': 'int',
    'Penalty Goals': 'int',
    '1 Point Field Goals': 'int',
    '2 Point Field Goals': 'int',
    'Total Points': 'int',
    'All Runs': 'int',
    'Line Breaks': 'int',
    'Line Break Assists': 'int',
    'Try Assists': 'int',
    'Line Engaged Runs': 'int',
    'Tackle Breaks': 'int',
    'Hit Ups': 'int',
    'Dummy Half Runs': 'int',
    'Dummy Half Run Metres': 'int',
    'One on One Steal': 'int',
    'Offloads': 'int',
    'Dummy Passes': 'int',
    'Passes': 'int',
    'Intercepts': 'int',
    'Kicks Defused': 'int',
    'Kicks': 'int',
    'Forced Drop Outs': 'int',
    'Bomb Kicks': 'int',
    'Grubbers': 'int',
    'Errors': 'int',
    'Handling Errors': 'int',
    'One on One Lost': 'int',
    'Penalties': 'int',
    'Ruck Infringements': 'int',
    'On Report': 'int',
    'Sin Bins': 'int',
    'Send Offs': 'int',
    'Goal Conversion Rate': 'int',
    'Receipts': 'int',
    'Tackles Made': 'int',                   
    'Missed Tackles': 'int',                  
    'Ineffective Tackles': 'int',
    '40/20': 'int',                           
    '20/40': 'int',                            
    'Cross Field Kicks': 'int',          
    'Kicked Dead': 'int',                 
    
    
    # Floats
    'Mins Played': 'float',
    'All Run Metres': 'float',
    'Kick Return Metres': 'float',
    'Post Contact Metres': 'float',
    'Play The Ball': 'float',
    'Average Play The Ball Speed': 'float',
    'Passes To Run Ratio': 'float',
    'Tackle Efficiency': 'float',
    'Kicking Metres': 'float',
    'Stint One': 'float',
    'Stint Two': 'float',
}