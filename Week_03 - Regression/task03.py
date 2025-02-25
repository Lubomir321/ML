def main():
    #What is the problem with always using R^2?
    #R² показва колко добре даден модел обяснява вариацията в зависимата променлива
    #въз основа на независимите променливи. Проблемът с R² е, че той винаги ще се увеличава
    #с добавянето на повече независими променливи към модела, дори ако тези променливи не
    #допринасят значително за обяснението на вариацията в зависимата променлива.
    #Това може да доведе до преоценка на модела и включване на нерелевантни променливи.


    #How does using Radj^2 help solve this problem?

    #R_adj², или коригираният коефициент на детерминация, регулира R² с оглед
    #на броя на включените променливи. Той наказва модела, когато се добавят
    #променливи, които не допринасят за подобряване на предсказателната сила на модела.


    #How could we calculate Radj^2 in Python?
    #from sklearn.metrics import r2_score

    #def adjusted_r2(y_true, y_pred, n, p):
    #    r2 = r2_score(y_true, y_pred)
    #    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    #y_pred - това са стойсностите предсказани от модела
    #n- боря на редовете или боря на пробите
    #p- боря на регресорите
    #https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2
    #https://www.investopedia.com/ask/answers/012615/whats-difference-between-rsquared-and-adjusted-rsquared.asp#toc-r-squared-vs-adjusted-r-squared-an-overview
    pass
if __name__ == '__main__':
    main()