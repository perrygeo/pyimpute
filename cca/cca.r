library(vegan)
library(labdsv)

data(varespec)
data(varechem)

vare.cca <- cca(varespec ~ Baresoil+Humdepth+pH+N+P+K+Ca+Mg+S+Al+Fe, data=varechem)
vare.cca
plot(vare.cca)
summary(vare.cca)

// See also http://ecology.msu.montana.edu/labdsv/R/labs/lab12/lab12.html