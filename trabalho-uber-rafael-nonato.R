# Bibliotecas

library(dplyr)
library(caret)
library(lubridate)
library(summarytools)
library(car)
library(rpart)
library(randomForest)
library(rattle)
library(rpart.plot)
library(caret)
library(Metrics)

#Endere?o do arquivo

# Lendo os arquivos no formato .csv (comma separated values)
DATA  <- read.csv("uber_nyc_enriched.csv", sep = ",", stringsAsFactors = T) 

str(DATA)
summary(DATA)

### Resumo de todos os dados disponiveis nos dados (completo)
DATA %>% group_by(borough) %>% 
summarise(`Pickups Totally` = sum(pickups)) %>% 
arrange(desc(`Pickups Totally`))

###aplicando os filtro para selecionar regi?o
DATA_TRAT =  DATA %>% filter(borough =="Brooklyn" 
                                  | borough =="Queens"
                                  | borough =="Manhattan")

summary(DATA_TRAT)
str(DATA_TRAT)

### Aplicando convers?o de dia da semana, hora e m?s
DATA_TRAT$borough= as.factor(as.character(DATA_TRAT$borough))
DATA_TRAT$pickup_dt = as_datetime(DATA_TRAT$pickup_dt)
DATA_TRAT = DATA_TRAT %>%
  mutate(mes = as.factor(as.character(month(pickup_dt, label = TRUE))))%>%
  mutate(weekday = as.factor(as.character(wday(pickup_dt, label= T))))%>%
  mutate(hora = hour(pickup_dt))

### convertendo dados de padr?es americanos para Brasileiro (Temperatura, Milhas e Polegadas)

## temperatura
DATA_TRAT$temp = (DATA_TRAT$temp-32) * (5/9)

## convertendo velocidade do vento
DATA_TRAT$spd = (DATA_TRAT$spd / 0.62137)

### convertendo velocidade do vento
DATA_TRAT$sd = (DATA_TRAT$sd / 0.39370)

summary(DATA_TRAT)

# Eliminando variavel pickups_dt
DATA_TRAT = DATA_TRAT [,-1]

#PROJETANDO OS DADOS
par(oma = c(2,3,0,0), mar = c(3,3,2,1), mfrow = c(2,2))
hist(DATA_TRAT$pickups,
      main = "Pickups Uber", cex.main = 2.0,
      xlab = "Qtda PickUps Uber",
      ylab= "frequency", cex.axis = 1.2,
      ylim = c(0,7000), col = 'blue', border = 'white')

barplot(table(DATA_TRAT$mes),
        main = "Months PickUps Uber", cex.main = 2.0,
        cex.names = 1.2,
        xlab = "Months",
        ylab= "frequency of pickups", cex.axis = 1.0,
        ylim = c(0,7000), col = 'blue', border = 'white')
        
# analise de demanda no periodo de feriado
par(oma = c(2,3,0,0), mar = c(3,3,2,1), mfrow = c(2,2)) # parametros graficos

# analise de demanda no periodo de feriado
boxplot(DATA_TRAT$pickups ~ DATA_TRAT$borough, data = DATA_TRAT,
        main = "Demand by Region", cex.main =1.5,
        xlab = "RRegion",
        ylab = "Uber Demand", cex.axis = 1.2,
        horizontal = F,
        ylim = c(0,4000), col = c('darkorange', 'darkorange', "darkorange"), border = "gray20")

boxplot(DATA_TRAT$pickups ~ DATA_TRAT$hday, data = DATA_TRAT,
        main = "Demand by Uber on holiday", cex.main =1.5,
        xlab = "holiday",
        ylab = "Uber demand", cex.axis = 1.2,
        horizontal = F,
        ylim = c(0,4000), col = c('darkorange', 'darkorange4'), border = "gray20")

#str(DATA_TRAT)

# EXTRAINDO AS VARIAVEIS EXPLICATIVA
X_REG <- DATA_TRAT[,-2]

# EXTRAINDO AS VARIAVEIS RESPOSTAS
Y_REG <- DATA_TRAT[,2, drop = F] # impedir que torne-se um vetor: drop = F

# Criando o objeto que constroi as dummies das variaveis categoricas
DUMMY_MODEL <- dummyVars(' ~ .', data = X_REG, sep = '_', fullRank = T)

X_REG <- as.data.frame(predict(DUMMY_MODEL, newdata = X_REG))


# 2) Correcao de assimetria na variavel resposta 
par(oma = c(2,3,0,0), mar = c(3,3,2,1), mfrow = c(2,2)) # parametros graficos

hist((DATA_TRAT$pickups), breaks = 10,
     main = 'Original distribuited', cex.axis= 1.2, cex.main= 1.2,
     ylim = c(0,10000), col = 'darkorange', border = 'darkorange4') 
hist(log10(DATA_TRAT$pickups), breaks = 10,
     main = expression(log(pickups)), cex.axis= 1.2, cex.main= 1.2,
     ylim = c(0,10000), col = 'darkorange', border = 'darkorange4') 
hist(log(DATA_TRAT$pickups), breaks = 10,
     main = expression(ln(pickups)), cex.axis= 1.2, cex.main= 1.2,
     ylim = c(0,6000), col = 'darkorange', border = 'darkorange4')
hist(sqrt(DATA_TRAT$pickups), breaks = 10,
     main = expression(sqrt(pickups)), cex.axis= 1.2, cex.main= 1.2,
     ylim = c(0,10000), col = 'darkorange', border = 'darkorange4') 
mtext(text="RESPONSE VARIABLE - ASSIMETRIC",
      side = 1, line = 0, outer = TRUE)
mtext(text="FREQUENCY (#)", side = 2, line = 0, outer = TRUE)


# Aplicando a raiz quadrada
summary(Y_REG)            # quantidade original
summary(sqrt(Y_REG))      # quantidade transformada
summary(sqrt(Y_REG)**2)   # voltando para o original
DATA_REG_PREPOC <- bind_cols(X_REG,Y_REG)

#####SEM USO #####
#summary(ln(Y_REG))
#Y_REG <- Y_REG %>% mutate(pickups = sqrt(pickups))


#Arvore de decis?o

DATA = DATA_REG_PREPOC

# 1) Divisao da base de modelagem em treino e teste com amostragem

set.seed(123) # garantindo reprodutibilidade da amostra

INDEX_TRAIN <- createDataPartition(DATA$pickups, p = 0.7, list = F)
TRAIN_SET <- DATA[INDEX_TRAIN, ] # base de desenvolvimento: 70%
TEST_SET  <- DATA[-INDEX_TRAIN,] # base de teste: 30%

# Avaliando a distribuicao da variavel resposta
summary(TRAIN_SET$pickups); summary(TEST_SET$pickups)

#--------------------------------------------------------------------------------#
# 2) Treino do algoritmo de arvore de regresssao

# aqui comecamos a arvore o mais completa possivel
MDL_FIT <- rpart(pickups ~.,
                 data = TRAIN_SET,
                 method = 'anova',
                 control = rpart.control(minbucket = 10, cp = 0.0015))
#??rpart.control

# saida da arvore
MDL_FIT
summary(MDL_FIT)

# avaliando a necessidade da poda da arvore
printcp(MDL_FIT)
window(MDL_FIT)
plotcp(MDL_FIT)


# aqui conseguimos podar a árvore controlando o cp que reduz o valor minimo do 
# erro, que é um parametro de controle
PARM_CTRL <- MDL_FIT$cptable[which.min(MDL_FIT$cptable[,"xerror"]),"CP"] ### comando para informar o poda automaticamente

# Podando a arvore
MDL_FIT.PRUNE <- prune(MDL_FIT, cp =0.0015)

# saida da árvore
# MDL_FIT.PRUNE
# summary(MDL_FIT.PRUNE)
# 
# #--------------------------------------------------------------------------------#
# # # 3) Plotando a arvore
# window(MDL_FIT.PRUNE)
# fancyRpartPlot(MDL_FIT.PRUNE)
# # 
# # #--------------------------------------------------------------------------------#
# # https://www.rdocumentation.org/packages/rpart.plot/versions/3.0.8/topics/rpart.plot
# rpart.plot(MDL_FIT.PRUNE,
#            cex = 0.5,
#            type = 3,
#            box.palette = "BuRd",
#            branch.lty = 3,
#            shadow.col ="gray",
#            nn = TRUE,
#            main = 'Regression Trees')
# # 
# # # https://www.rdocumentation.org/packages/rpart.plot/versions/3.0.8/topics/rpart.plot
# rpart.plot(MDL_FIT.PRUNE,
#            type = 3,
#            cex = 0.5,
#            clip.right.labs = FALSE,
#            branch = .4,
#            box.palette = "BuRd",       # override default GnBu palette
#            main = 'Regression Trees')
#--------------------------------------------------------------------------------#
# 4) Realizando as predicoes

# Valor de AMOUNT pela arvore regressao com maior desenvolvimento
Y_VAL_TRAIN <- predict(MDL_FIT.PRUNE) 
Y_VAL_TEST  <- predict(MDL_FIT.PRUNE, newdata = TEST_SET)

#--------------------------------------------------------------------------------#
# 5) Avaliando a performance dos modelos e existencia de overfitting

# Arvore
postResample(pred = Y_VAL_TRAIN, obs = TRAIN_SET$pickups)
postResample(pred = Y_VAL_TEST,  obs = TEST_SET$pickups)

# sinais de overfitting entre as amostras de treino e teste? 
MDL_FINAL <- MDL_FIT.PRUNE

#--------------------------------------------------------------------------------#
# 6) Importancia das variaveis (Modelo final)

# o algoritmo de arvore tambem possui uma saida com a importancia das variaveis
round(MDL_FINAL$variable.importance, 3)

#--------------------------------------------------------------------------------#
# 7) Inspecao dos valores previstos vs observados (modelo final)

# Convertendo a variavel para unidade original
RESULT_TRAIN <- data.frame(AMOUNT_OBS  = TRAIN_SET$pickups,
                           AMOUNT_PRED = Y_VAL_TRAIN) %>%
  mutate(RESIDUO = AMOUNT_PRED - AMOUNT_OBS)

RESULT_TEST  <- data.frame(AMOUNT_OBS  = TEST_SET$pickups,
                           AMOUNT_PRED = Y_VAL_TEST) %>%
  mutate(RESIDUO = AMOUNT_PRED - AMOUNT_OBS)

# Plotando os resultados
#window()
layout(matrix(c(1,2,3,4,3,4), nrow = 3, ncol = 2, byrow = TRUE))
par(oma = c(1,1,0,0),  mar = c(5,5,2,1))

hist(RESULT_TRAIN$RESIDUO, breaks = 12, xlim = c(-2000,2000),
     main = 'training sample', cex.main = 1.2, 
     xlab = 'RESIDUAL', ylab = 'FREQUENCY (#)', cex.axis = 1.2,  
     col = 'darkorange', border = 'darkorange4')

hist(RESULT_TEST$RESIDUO, breaks = 12, xlim = c(-2000,2000),
     main = 'training test', cex.main = 1.2, 
     xlab = 'RESIDUAL', ylab = 'FREQUENCY (#)', cex.axis = 1.2,  
     col = 'darkorange', border = 'darkorange4')

plot(RESULT_TRAIN$AMOUNT_PRED,RESULT_TRAIN$AMOUNT_OBS,
     #main = 'Amostra de Treino', cex.main = 1.2, 
     xlab = 'Uber Demand(provided)', ylab = 'Uber Demand(observed)',
     cex.axis = 1.2, pch = 19, cex = 0.5, ylim = c(0,6000),
     col = 'darkorange')
abline(lm(AMOUNT_OBS ~ AMOUNT_PRED, data = RESULT_TRAIN), 
       col = 'red', lwd = 3)
abline(0, 1, col = 'blue', lwd = 3, lty = "dashed")

plot(RESULT_TEST$AMOUNT_PRED, RESULT_TEST$AMOUNT_OBS, 
     #main = 'Amostra de Teste', cex.main = 1.2, 
     xlab = 'Uber Demand(provided)', ylab = 'Uber Demand(observed)',
     cex.axis = 1.2, pch = 19, cex = 0.5, ylim = c(0,6000),
     col = 'darkorange')
abline(lm(AMOUNT_OBS ~ AMOUNT_PRED, data = RESULT_TEST), 
       col = 'red', lwd = 3)
abline(0, 1, col = 'blue', lwd = 3, lty = "dashed")

# graphics.off()
# rm(list = ls())

#### ---------fim-------------
###
###

#--------------------------------------------------------------------------------#
# RANDOM FOREST PARA REGRESSAO
#--------------------------------------------------------------------------------#
# 0) Lendo a base de dados

# Selecionando o working directory

DATA = DATA_REG_PREPOC

#--------------------------------------------------------------------------------#
# 1) Divisao da base de modelagem em treino e teste com amostragem

set.seed(123) # garantindo reprodutibilidade da amostra

INDEX_TRAIN <- createDataPartition(DATA$pickups, p = 0.7, list = F)
TRAIN_SET <- DATA[INDEX_TRAIN, ] # base de desenvolvimento: 70%
TEST_SET  <- DATA[-INDEX_TRAIN,] # base de teste: 30%

# Avaliando a distribuicao da variavel resposta
summary(TRAIN_SET$pickups);
summary(TEST_SET$pickups)

#--------------------------------------------------------------------------------#
# 2) Treino do algoritmo do random forest para regresssao  RETIRAR OS COMENTARIOS - MAQUINA VAI EXPLODIR

MDL_FIT <- randomForest(pickups ~ .,  ############## COMENTAR
                        data = TRAIN_SET,
                        importance = T,
                        mtry       =24,
                        nodesize   = 3,
                        ntree      = 300)
#?randomForest
#saida do modelo
MDL_FIT ############## COMENTAR
plot(MDL_FIT, main = 'Out-of-bag error')  ############## COMENTAR\

# acessando cada arvore individualmente
# #windows()
getTree(MDL_FIT, k = 300, labelVar=TRUE)  ############## COMENTAR

# a partir de 400 arvores o erro estabiliza e nao ha mais melhoria
##MDL_FIT1 <- randomForest(pickups ~ .,
##                        data = TRAIN_SET,
  ##                      importance = T,
    ##                    mtry       = 24,
      ##                  nodesize   = 3, 
        ##                ntree      = 300)


#--------------------------------------------------------------------------------#
# 3) Realizando as predicoes

# Valor de AMOUNT pelo random forest 
Y_VAL_TRAIN <- predict(MDL_FIT) 
Y_VAL_TEST  <- predict(MDL_FIT, newdata = TEST_SET)

#--------------------------------------------------------------------------------#
# 4) Avaliando a performance dos modelos e existencia de overfitting

# Rando forest para regress?o
postResample(pred = Y_VAL_TRAIN, obs = TRAIN_SET$pickups)
postResample(pred = Y_VAL_TEST,  obs = TEST_SET$pickups)


# sinais de overfitting entre as amostras de treino e teste? 
MDL_FINAL <- MDL_FIT

summary(MDL_FINAL)
#--------------------------------------------------------------------------------#
# 5) Importancia das variaveis (Modelo final)
#varImp(MDL_FINAL)

# o pacote random forest tamb?m possui uma forma de ver a importancia das 
# variaveis
varImpPlot(MDL_FINAL, sort= T, main = 'Importancia das Variaveis')

#--------------------------------------------------------------------------------#
# 6) Inspecao dos valores previstos vs observados (modelo final)

# Convertendo a variavel para unidade original
RESULT_TRAIN <- data.frame(AMOUNT_OBS  = TRAIN_SET$pickups,
                           AMOUNT_PRED = Y_VAL_TRAIN) %>%
  mutate(RESIDUO = AMOUNT_PRED - AMOUNT_OBS)

RESULT_TEST  <- data.frame(AMOUNT_OBS  = TEST_SET$pickups,
                           AMOUNT_PRED = Y_VAL_TEST) %>%
  mutate(RESIDUO = AMOUNT_PRED - AMOUNT_OBS)

# Plotando os resultados
#window()
layout(matrix(c(1,2,3,4,3,4), nrow = 3, ncol = 2, byrow = TRUE))
par(oma = c(1,1,0,0),  mar = c(5,5,2,1))


hist(RESULT_TRAIN$RESIDUO, breaks = 25, xlim = c(-2000,2000),
     main = 'Training sample', cex.main = 1.2, 
     xlab = 'RESIDUAL', ylab = 'FREQIENCY (#)', cex.axis = 1.2,  
     col = 'darkorange', border = 'darkorange4')

hist(RESULT_TEST$RESIDUO, breaks = 25, xlim = c(-2000,2000),
     main = 'Training Test', cex.main = 1.2, 
     xlab = 'RESIDUAL', ylab = 'FREQIENCY (#)', cex.axis = 1.2,  
     col = 'darkorange', border = 'darkorange4')

plot(RESULT_TRAIN$AMOUNT_OBS,RESULT_TRAIN$AMOUNT_PRED,
     #main = 'Amostra de Treino', cex.main = 1.2, 
     xlab = 'Uber Demand(Observed)', ylab = 'Uber Demand(provided)',
     cex.axis = 1.2, pch = 19, cex = 0.5, ylim = c(0,6000),
     col = 'darkorange')
abline(lm(AMOUNT_PRED ~ AMOUNT_OBS, data = RESULT_TRAIN), 
       col = 'firebrick', lwd = 3)
abline(0, 1, col = 'blue', lwd = 3, lty = "dashed")

plot(RESULT_TEST$AMOUNT_OBS, RESULT_TEST$AMOUNT_PRED, 
     #main = 'Amostra de Teste', cex.main = 1.2, 
     xlab = 'Uber Demand(Observed)', ylab = 'Uber Demand(provided)',
     cex.axis = 1.2, pch = 19, cex = 0.5, ylim = c(0,6000),
     col = 'darkorange')
abline(lm(AMOUNT_PRED ~ AMOUNT_OBS, data = RESULT_TEST), 
       col = 'firebrick', lwd = 3)
abline(0, 1, col = 'blue', lwd = 3, lty = "dashed")

### Analise Bi variada

#-----------------------------------------------------
# An?lises:
# - Analise a m?trica do R2 do modelo para cada bairro separado:
# o modelo captura avariabilidade dos dados de forma parecida entre as 3 regi?es?;

par(mfrow = c(1,1))

# avaliando R2 por bairro

treino <- TRAIN_SET
teste <- TEST_SET

treino$pred <- predict(MDL_FIT) 
teste$pred  <- predict(MDL_FIT, newdata = teste)

#Manhattan

Manhattan_train <- treino %>% filter(borough_Manhattan==1)
Manhattan_test<- teste %>% filter(borough_Manhattan==1)


Manhattan_test %>% summarise(`Total Pickups` = sum(pickups)) %>% arrange(desc(`Total Pickups`))


postResample(pred = Manhattan_train$pred, obs = Manhattan_train$pickups)
postResample(pred = Manhattan_test$pred,  obs = Manhattan_test$pickups)


#Queens

Queens_train <- treino %>% filter(borough_Queens==1)
Queens_test<- teste %>% filter(borough_Queens==1)


Queens_test %>% summarise(`Total Pickups` = sum(pickups)) %>% 
  arrange(desc(`Total Pickups`))


postResample(pred = Queens_train$pred, obs = Queens_train$pickups)
postResample(pred = Queens_test$pred,  obs = Queens_test$pickups)

#Brooklyn

Brooklyn_train <- treino %>% filter(borough_Queens==0 & borough_Manhattan ==0)
Brooklyn_teste<- teste %>% filter(borough_Queens==0 & borough_Manhattan ==0)


Brooklyn_teste %>% summarise(`Total Pickups` = sum(pickups)) %>% 
  arrange(desc(`Total Pickups`))


postResample(pred = Brooklyn_train$pred, obs = Brooklyn_train$pickups)
postResample(pred = Brooklyn_teste$pred,  obs = Brooklyn_teste$pickups)

# --------------------------
#calculando variacao
# --------------------------

teste$var <- (teste$pred - teste$pickups)/teste$pickups
#plotando variacao
#windows()
hist(teste$var, breaks = 10000,
     xlim = c(-1,1),
     main = 'Varia??o Previsto x Observado', cex.main = 1.2, 
     xlab = 'RESIDUAL', ylab = 'FREQUENCIA (#)', cex.axis = 1.2,  
     col = 'dodgerblue', border = 'dodgerblue4')

teste$acerto <- ifelse(teste$var >.1,0,
                          ifelse(teste$var< -.1,0, 1))
table(teste$acerto)

# SUP1 >75%,
# SUP2 75%-50%,
# SUP3 50%-25%,
# SUP4 25%-10%,
# 
# ACERTO (SUP10% -- SUB10%)
# 
# SUB1 >75%,
# SUB2 75%-50%,
# SUB3 50%-25%,
# SUB4 25%-10%,

teste$faixas <- ifelse(teste$acerto == 1, 'ACERTO',
                          ifelse(teste$var >.75,'SUP1',
                                 ifelse(teste$var > .5,'SUP2',
                                        ifelse(teste$var > .25,'SUP3',
                                               ifelse(teste$var > .1, 'SUP4',
                                                      ifelse(teste$var < -.75,'SUB1',
                                                             ifelse(teste$var < -.5, 'SUB2',
                                                                    ifelse(teste$var < -.25, 'SUB3','SUB4'))))))))

table(teste$faixas)

Mht_test<- teste %>% filter(borough_Manhattan==1)
Queens_test<- teste %>% filter(borough_Queens==1)
Brooklyn_test<- teste %>% filter(borough_Manhattan==0 & borough_Queens ==0)

table(Brooklyn_test$faixas)
table(Queens_test$faixas)
table(Mht_test$faixas)

