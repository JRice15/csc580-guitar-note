import math
import warnings
import numpy as np

from midi_to_fret import MIDI_TO_FRET


MAX_STRETCH = 3

# Given the predictions (midi values) from the model
# and a mapping of midi values to all possible strings and frets to play that midi value
# create the tablature

# midi_to_fret: dict{ midi_note: (string, fret) }

# make_fretting - given list of midi values for any time step, and the list of the previous frettings, return the next cheapest fretting
# midi_vals: list of ints
# prev fretting: list of tuples (string #, fret #) where 1 is the low E, 0 is open string

# if not prev input, find location of root note, then for remaining midi vals find closest
# if prev inputm find location of prev input root note, find closest next root note
def find_fretting_old(midi_vals, prev_midi_vals, prev_fretting):
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


def min_dist(str_and_fret, prev_fretting):
    """
    get shortest distance from a string-fret combo to a previous fretting
    """
    min_dist = 100000
    for prev in prev_fretting:
        str_dist = abs(prev[0] - str_and_fret[0])
        fret_dist = abs(prev[1] - str_and_fret[1])
        # dist = math.sqrt((str_dist**2)+(fret_dist**2))
        dist = str_dist + fret_dist
        min_dist = min(dist, min_dist)
    return min_dist


def find_fretting(midi, prev_fretting=None):
    if len(midi) == 0:
        return []
    midi = sorted(midi)
    # if there are 6 midi notes, the chord is entirely specified and there's only one way to play it
    if prev_fretting is None or len(midi) == 6:
        # sort high to low
        midi = midi[::-1]
        # get lowest possible fretting for highest note
        highest_note_fretting = MIDI_TO_FRET[midi[0]][-1]

        fretting = [highest_note_fretting]
        for m in midi[1:]:
            # get already used strings, and min/max fret range
            used_strings = [f[0] for f in fretting]
            frets = [f[1] for f in fretting]
            min_fret = min(frets)
            max_fret = max(frets)
            options = MIDI_TO_FRET[m]
            options = [opt for opt in options if (opt[1] >= min_fret - MAX_STRETCH) and (opt[1] <= max_fret + MAX_STRETCH) and (opt[0] not in used_strings)]
            if not len(options):
                warnings.warn("No option for " + str(m))
            fretting.append(options[-1])
        return fretting[::-1]
    
    else:
        # highest note min fret
        min_fret = MIDI_TO_FRET[midi[-1]][-1][1]

        # get lowest note options
        options = MIDI_TO_FRET[midi[0]]
        options = [opt for opt in options if (opt[1] >= min_fret - MAX_STRETCH) ]
        opt_distances = [min_dist(opt, prev_fretting) for opt in options]
        lowest_note_fretting = options[np.argmin(opt_distances)]

        fretting = [lowest_note_fretting]
        # do the rest from the top down
        for m in midi[1:][::-1]:
            # get already used strings, and min/max fret range
            used_strings = [f[0] for f in fretting]
            frets = [f[1] for f in fretting]
            min_fret = min(frets)
            max_fret = max(frets)
            options = MIDI_TO_FRET[m]
            options = [opt for opt in options if (opt[1] >= min_fret - MAX_STRETCH) and (opt[1] <= max_fret + MAX_STRETCH) and (opt[0] not in used_strings)]
            if not len(options):
                warnings.warn("gave up for {} {}".format(midi, prev_fretting))
                return find_fretting(midi, None)
            opt_distances = [min_dist(opt, prev_fretting) for opt in options]
            fretting.append(options[np.argmin(opt_distances)])
        return sorted(fretting)



if __name__ == "__main__":
    from tab_printing import Tabulator

    chords = [
        [48, 55, 60, 64], # C bar
        [40, 47, 52, 55, 59, 64], # Em low
        [50, 57, 62, 66], # D flex
        [47, 52, 59, 64, 67, 71], # high em bar
        [48, 52, 55, 60], # C no bar
    ]

    t = Tabulator()
    prev = None
    for c in chords:
        fretting = find_fretting(c, prev_fretting=prev)
        t.add_timestep(fretting)
        prev = fretting

    t.print_out()


