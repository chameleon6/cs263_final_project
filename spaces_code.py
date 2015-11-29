chunk_lens = map(lambda (x,y): x-y, zip(np.append(starts[1:], [len(data)]), starts))
spacinesses = np.array([x+y for (x,y) in zip([0] + chunk_lens, chunk_lens)])
space_thresh = sorted(spacinesses)[int(0.5 * len(spacinesses))]
space_guesses_from_time = []

for i,v in enumerate(spacinesses):
    if v > space_thresh:
        should_append = True
        if len(space_guesses_from_time) > 0 and space_guesses_from_time[-1] == i-1:
            if v < spacinesses[i-1]:
                should_append = False
            else:
                space_guesses_from_time = space_guesses_from_time[:-1]
        if should_append:
            space_guesses_from_time.append(i)

#N, M = len(data) - 500000, len(data)
N, M = 0, 1000000
ii = np.zeros(len(data))
ii[starts] = 1
ii2 = np.zeros(len(data))
ii2[ends] = 1

plot_ind_start = None
plot_ind_end = None
for i,v in enumerate(starts):
    if v > N and plot_ind_start == None:
        plot_ind_start = i
    if v > M:
        plot_ind_end = i+1
        break

ii_space = np.zeros(len(data))

typed_words = text.split(' ')
typed_lens = map(len, typed_words)
guessed_lens = [x-y-1 for (x,y) in zip(space_guesses_from_time, [-1] + space_guesses_from_time)]
typed_lens = np.array(typed_lens)
guessed_lens = np.array(guessed_lens)
print "actual word lens:"
print typed_lens
print "guessed lens:"
print guessed_lens

typed_word_inds, guessed_word_inds, good_word_lens = longest_common_subseq(typed_lens, guessed_lens)
starts = np.array(starts)[space_guesses_from_time]
supervised_train_data = []

for t,g,l in zip(typed_word_inds, guessed_word_inds, good_word_lens):
    real_word = typed_words[t]
    this_chunk = space_guesses_from_time[g-1]+1 if g > 0 else 0
    print real_word, this_chunk
    for i in range(l):
        supervised_train_data.append((real_word[i], chunks[this_chunk + i]))

for i in starts:
    ii_space[i] = 1

