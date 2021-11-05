import math
from midi_to_fret import midi_to_fret

# Given the predictions (midi values) from the model
# and a mapping of midi values to all possible strings and frets to play that midi value
# create the tablature

# midi_to_fret: dict{ midi_note: (string, fret) }

# make_fretting - given list of midi values for any time step, and the list of the previous frettings, return the next cheapest fretting
# midi_vals: list of ints
# prev fretting: list of tuples (string #, fret #) where 1 is the low E, 0 is open string

# if not prev input, find location of root note, then for remaining midi vals find closest
# if prev inputm find location of prev input root note, find closest next root note
def find_fretting(midi_vals, prev_fretting):
    new_fretting = []
    if prev_fretting is None:
        # first fretting, default to start closest to fret 0
        first = midi_vals[0]
        cur_min = midi_to_fret[first][0]

        # loop through the rest of the possible frettings, choose the minimum
        for tupl in midi_to_fret[first]:
            if tupl[1] < cur_min[1]:
                cur_min = tupl
        
        new_fretting.append(cur_min)
        prev = cur_min

        # first fretting established, calculate next cheapest frettings
        for f in midi_vals[1:]:
            min_dist = 10000
            possible_frettings = midi_to_fret[f]
            for str_and_fret in possible_frettings:
                # print(f, str_and_fret)
                str_dist = prev[0] - str_and_fret[0]
                fret_dist = prev[1] - str_and_fret[1]
                dist = math.sqrt((str_dist**2)+(fret_dist**2))
                if dist < min_dist:
                    min_dist = dist
                    cur_min = str_and_fret
            new_fretting.append(cur_min)
            prev = (cur_min)
    
    return new_fretting

midi_vals = [40, 41, 45, 46]

# find_fretting(midi_vals, None)
print(find_fretting(midi_vals, None))
# print (find_fretting(scale, None))

