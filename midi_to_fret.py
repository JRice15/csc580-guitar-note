lowest_note = [40,45,50,55,59,64]
string_notes = [[] for i in range(6)]
for string in range(0,6):
    for fret in range(0,19):
        string_notes[string].append(lowest_note[string] + fret)

for i in range(0,6):
    print("String " + str(i) + ":")
    print(string_notes[i])

for note in range(40, 83):
    frettings = []
    for string in range(6):
        try:
            fret = string_notes[string].index(note)
            frettings.append((string+1, fret))
        except ValueError:
            pass
    print(str(note) + " : " + str(frettings) + ",")


# #1st string is low E
# #6th string is high E
# 40 to 82

midi_to_fret = {
    40 : [(1, 0)],
    41 : [(1, 1)],
    42 : [(1, 2)],
    43 : [(1, 3)],
    44 : [(1, 4)],
    45 : [(1, 5), (2, 0)],
    46 : [(1, 6), (2, 1)],
    47 : [(1, 7), (2, 2)],
    48 : [(1, 8), (2, 3)],
    49 : [(1, 9), (2, 4)],
    50 : [(1, 10), (2, 5), (3, 0)],
    51 : [(1, 11), (2, 6), (3, 1)],
    52 : [(1, 12), (2, 7), (3, 2)],
    53 : [(1, 13), (2, 8), (3, 3)],
    54 : [(1, 14), (2, 9), (3, 4)],
    55 : [(1, 15), (2, 10), (3, 5), (4, 0)],
    56 : [(1, 16), (2, 11), (3, 6), (4, 1)],
    57 : [(1, 17), (2, 12), (3, 7), (4, 2)],
    58 : [(1, 18), (2, 13), (3, 8), (4, 3)],
    59 : [(2, 14), (3, 9), (4, 4), (5, 0)],
    60 : [(2, 15), (3, 10), (4, 5), (5, 1)],
    61 : [(2, 16), (3, 11), (4, 6), (5, 2)],
    62 : [(2, 17), (3, 12), (4, 7), (5, 3)],
    63 : [(2, 18), (3, 13), (4, 8), (5, 4)],
    64 : [(3, 14), (4, 9), (5, 5), (6, 0)],
    65 : [(3, 15), (4, 10), (5, 6), (6, 1)],
    66 : [(3, 16), (4, 11), (5, 7), (6, 2)],
    67 : [(3, 17), (4, 12), (5, 8), (6, 3)],
    68 : [(3, 18), (4, 13), (5, 9), (6, 4)],
    69 : [(4, 14), (5, 10), (6, 5)],
    70 : [(4, 15), (5, 11), (6, 6)],
    71 : [(4, 16), (5, 12), (6, 7)],
    72 : [(4, 17), (5, 13), (6, 8)],
    73 : [(4, 18), (5, 14), (6, 9)],
    74 : [(5, 15), (6, 10)],
    75 : [(5, 16), (6, 11)],
    76 : [(5, 17), (6, 12)],
    77 : [(5, 18), (6, 13)],
    78 : [(6, 14)],
    79 : [(6, 15)],
    80 : [(6, 16)],
    81 : [(6, 17)],
    82 : [(6, 18)]
} 