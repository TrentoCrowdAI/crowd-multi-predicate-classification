data <- matrix(c(31.19, 84.29, 92.01, 
                 6.28, 49.38, 70.66, 
                 11.07, 82.80, 90.57,
                 1.56, 47.97, 69.70),
               ncol=3,byrow=TRUE)
colnames(data) <- c("Rd_cheaters","Smart_cheaters","Workers")
rownames(data) <- c("trsh: 75%\npapers: 4",
                   "trsh: 100%\npapers: 4",
                   "trsh: 75%\npapers: 6",
                   "trsh: 100%\npapers: 6")

barplot(t(data), main="Passed quiz populatinos vs. parameters settings",
        col=c("gray","darkmagenta", "darkolivegreen1"), beside=TRUE,
        legend.text = c('rand cheaters','smart cheaters','trusted workers'),
        ylab = "Proportion from a population", ylim = c(0, 100), border = "dark blue",
        args.legend = list(x ='topright', bty='n'))
