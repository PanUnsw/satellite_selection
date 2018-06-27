# star_selection
select star

## NEXT:
- [x] Make star id consistant in input data
- [x] Integrate loss of positive number constrain
- [x] Add positive num as input
- [x] Output error cases
- [ ] Check valid padding
- [ ] Analyze the efficiency of num_pos in multiple num_pos case
- [ ] Cut star num to use larger conv size
- [ ] Try to use num_pos more efficiently at the very early stage
- [x] Instead of classify to two classes, try to classify to one class. Selecct the num_pos maximum of probability to this class.
- [x] Shuffle input order
- [ ] Think of drop out
- [ ] Integrate loss of time consistant
- [x] Use weighted loss, the loss for missing stars are smaller
- [ ] Figure out some way to sort all the satelites
