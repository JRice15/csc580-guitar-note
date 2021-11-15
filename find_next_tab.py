import math
from midi_to_fret import MIDI_TO_FRET

# Given the predictions (midi values) from the model
# and a mapping of midi values to all possible strings and frets to play that midi value
# create the tablature

# midi_to_fret: dict{ midi_note: (string, fret) }

# make_fretting - given list of midi values for any time step, and the list of the previous frettings, return the next cheapest fretting
# midi_vals: list of ints
# prev fretting: list of tuples (string #, fret #) where 1 is the low E, 0 is open string

# if not prev input, find location of root note, then for remaining midi vals find closest
# if prev inputm find location of prev input root note, find closest next root note
def find_fretting(midi_vals, prev_midi_vals, prev_fretting):
    new_fretting = []
    if prev_fretting is None:
        # first fretting, default to start at root note closest to fret 0
        min_index = midi_vals.index(min(midi_vals))
        first = midi_vals[min_index]

        cur_min = MIDI_TO_FRET[first][0]
        # loop through the rest of the possible frettings, choose the minimum
        for tupl in MIDI_TO_FRET[first]:
            if tupl[1] < cur_min[1]:
                cur_min = tupl
        
        new_fretting.append(cur_min)
        prev = cur_min

        # first fretting established, calculate next cheapest frettings
        for f in midi_vals[1:]:
            min_dist = 10000
            possible_frettings = MIDI_TO_FRET[f]
            for str_and_fret in possible_frettings:
                # print(f, str_and_fret)
                str_dist = prev[0] - str_and_fret[0]
                fret_dist = prev[1] - str_and_fret[1]
                dist = math.sqrt((str_dist**2)+(fret_dist**2))
                if dist < min_dist:
                    min_dist = dist
                    cur_min = str_and_fret
            new_fretting.append(cur_min)
            prev = cur_min
    
    else:
        # finding the prev root midi value
        prev_min_index = prev_midi_vals.index(min(prev_midi_vals))
        prev_root_midi = prev_midi_vals[prev_min_index]

        # finding the prev fretting used for the prev root midi value
        for frettings in prev_fretting:
            # print(frettings)
            if frettings in MIDI_TO_FRET[prev_root_midi]:
                prev_root_fretting = frettings
                break

        # finding the current root midi value
        curr_min_index = midi_vals.index(min(midi_vals))
        curr_root_midi = midi_vals[curr_min_index]

        # finding the root fretting closest to prev root fretting
        root_midi_frets = MIDI_TO_FRET[curr_root_midi]
        # print(root_midi_frets)
        min_dist = 10000
        for str_and_fret in root_midi_frets:
            str_dist = prev_root_fretting[0] - str_and_fret[0]
            fret_dist = prev_root_fretting[1] - str_and_fret[1]
            dist = math.sqrt((str_dist**2)+(fret_dist**2))
            if dist < min_dist:
                min_dist = dist
                cur_min = str_and_fret
                # print(cur_min)
        new_fretting.append(cur_min)
        prev = cur_min

        # loop through remaining midi vals
        # find closest fretting to current root fretting
        for f in midi_vals:
            if f != curr_root_midi:
                min_dist = 10000
                possible_frettings = MIDI_TO_FRET[f]
                for str_and_fret in possible_frettings:
                    # print(f, str_and_fret)
                    str_dist = prev[0] - str_and_fret[0]
                    fret_dist = prev[1] - str_and_fret[1]
                    dist = math.sqrt((str_dist**2)+(fret_dist**2))
                    if dist < min_dist:
                        min_dist = dist
                        cur_min = str_and_fret
                new_fretting.append(cur_min)
                prev = cur_min

    return new_fretting

Em_midi_vals = [40, 47, 52, 55, 59, 64]
D_midi_vals = [50, 57, 62, 66]


# find_fretting(midi_vals, None)
Em_frettings = find_fretting(Em_midi_vals, None, None)
print(Em_frettings)
D_frettings = find_fretting(D_midi_vals, Em_midi_vals, Em_frettings)
print(D_frettings)

# print (find_fretting(scale, None))

