## API translator

I want to have some kind of quick translator layer for famous genomic tools like regenie and plink. like we can do

g api-plink and then plik flags
or
g api-regenie and then regenie flags

also can have python api like

from g import api_plink as g

g(
    and here arguments with the same names as plink
)

also from g import api_hail, api_regenie

## UX documentation

Also i want my app to have great ui and ux, in case of bioinformatics ux is partly documentation. i think neat place for documentation is github wiki

i want to have every flag explained. if flag is actually responsible for some complex math or algorithm - i want it mentioned and linked

Also would be nice to have examples or case studies for what a parameter could mean

When thinking of this i realized that to have this in place, we have to have documentation on computational genetics itself in a repo, like what the fuck is even firth, how our app decides to use it. what are dosages. what are different file formats, what are they. and then i can have parameters documentation linked to that. 

Probably we should write documentation for app and if you need to know algorithm to use it correctly, we will write explanation in wiki

Also want to have something like no stupid question policy. If person asks question on something, it means that we either did not write it in documentation, or were bad at explaining what he needs, or documentation was in non obvious place. we will try to address each question and keep library of those in github issues to make them searchable.

## CI to check reliability

I want to have some kind of ci that will run every month and check for if app still does its functionality, so basically tests, but also some integration tests to run on actual files. This should prevent software from rotting

## Versioning

For software like this there can actually be breaking changes. breaking change can be change in api so it no longer works in a pipeline. it can be change in flag name, or change in output format, or change in output column names. new minor version will probably indicate new features, or something considerably new, need to think about it.

## Negative log p-value

I want app to compute p values in negative log10, i believe that only this makes sense. it will be easier for plotting and will allow to change dtype to bfloat16 for even faster code.