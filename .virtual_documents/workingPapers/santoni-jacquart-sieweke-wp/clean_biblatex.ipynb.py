from companionCode.handleBibs import clean_bib


# file with BibLaTeX items to source
in_f = 'review/sampled_studies.bib'
# destnation file
out_f = 'manuscript/references/sampled_studie.bib'
# common words
common_words = [' A ', ' An ', ' And ', ' Of ', ' The ', ' In ', 
                ' On ', ' From ', ' Over ', ' To ',  ' That ',
                ' Which ', ' For ', ' With ', ' Only ', ' Among ',
                ' By ', ' As ', ' Whether ', ' Or ', ' At ', 
                ' Versus ', ' -To- ', 'Title = ']
# run function
clean_bib(in_f=in_f, out_f=out_f, common_words=common_words)



