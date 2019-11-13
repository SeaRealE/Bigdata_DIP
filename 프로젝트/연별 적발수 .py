import pandas as pd

def fileName(listInput):
    string = "("
    for tmp in listInput.columns:
        if tmp == listInput.columns[-1]:
            string += tmp
        else :
            string += tmp+"_"
    string += ")"

    return string

# EXCEL -> '위반장소명' 의 표시형식을 '대구 중구 @'로 바꿔주기
# 년도만 바꾸기 
year = 2018

oldFile = pd.read_csv('./'+str(year)+'년 불법 주정차 단속 현황(대구중구).csv', encoding="cp949")        
coordinate = pd.read_csv('./위경도추출.csv', encoding="cp949")                             
newFile = pd.merge(oldFile, coordinate,how='left',on='위반장소명')

indexList = []

## 대구 포함 위경도 
#36.011326, 128.757661
#35.607218, 128.373762
for index in newFile.index:
    if newFile.loc[index, '위도'] < 35.607218 or newFile.loc[index, '위도'] > 36.011326:
        indexList.append(index)
    if newFile.loc[index, '경도'] < 128.373762 or newFile.loc[index, '경도'] > 128.757661:
        indexList.append(index)

newFile = newFile.drop(indexList)
newFile = newFile.reset_index().drop(["index"],axis=1)
        
newFile.to_csv(str(year)+"년 위경도추가.csv", mode='w', index=False, encoding="cp949")

# 적발수 관련 csv 파일 생성
###############################################
countFile = pd.DataFrame(newFile.groupby(by=['위반장소명','위도','경도']).count())
countFile = countFile1.reset_index() 
countFile.columns = ['위반장소명','위도','경도','적발수','','']
countFile = countFile[['위반장소명','위도','경도','적발수']]

countFile.to_csv(str(year)+"년 적발수"+fileName(countFile)+".csv", mode='w', index=False, encoding="cp949")




