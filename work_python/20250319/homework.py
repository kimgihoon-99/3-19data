# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 12:40:20 2025

@author: Admin
"""
import pandas as pd
import numpy as np
import folium

df = pd.read_csv("./bc_card_data/201906.csv")

# 서울시에 거주하는 고객 수
seoul_customers = df[df['CSTMR_MEGA_CTY_NM'] == '서울특별시'].shape[0]

# 서울시에 거주하지 않는 고객 수
non_seoul_customers = df[df['CSTMR_MEGA_CTY_NM'] != '서울특별시'].shape[0]

# 분석 결과 출력
print("서울시에 거주하는 고객 수:", seoul_customers, "명")
print("서울시에 거주하지 않는 고객 수:", non_seoul_customers, "명")

# 서울시 거주 고객의 총 소비액 계산
seoul_spending = df[df['CSTMR_MEGA_CTY_NM'] == '서울특별시']['AMT'].sum()

# 서울시 비거주 고객의 총 소비액 계산
non_seoul_spending = df[df['CSTMR_MEGA_CTY_NM'] != '서울특별시']['AMT'].sum()

# 분석 결과 출력
print("서울시 거주 고객의 총 소비액:", f"{seoul_spending:,}원")
print("서울시 비거주 고객의 총 소비액:", f"{non_seoul_spending:,}원")

# 전체 고객의 총 소비액 계산
total_spending = df['AMT'].sum()

# 분석 결과 출력
print("전체 고객의 총 소비액:", f"{total_spending:,}원")


# 성별 소비액 계산 (SEX_CTGO_CD: 1=남성, 2=여성)
male_spending = df[df['SEX_CTGO_CD'] == 1]['AMT'].sum()
female_spending = df[df['SEX_CTGO_CD'] == 2]['AMT'].sum()

# 분석 결과 출력
print("남성 고객의 총 소비액:", f"{male_spending:,}원")
print("여성 고객의 총 소비액:", f"{female_spending:,}원")

# 카드 이용 건수 합산
total_transactions = df['CNT'].sum()

# 분석 결과 출력
print("전체 카드 이용 건수:", f"{total_transactions:,}건")

# TP_BUZ_NM이 '편 의 점'으로 정확히 일치하는 행의 소비액 계산
convenience_store_spending = df[df['TP_BUZ_NM'] == '편 의 점']['AMT'].sum()

# 분석 결과 출력
print("편의점(편 의 점)의 총 소비액:", f"{convenience_store_spending:,}원")


# 조건에 맞는 데이터 필터링 (편의점 & 강남구)
gangnam_convenience_spending = df[
    (df['TP_BUZ_NM'] == '편 의 점') & (df['CTY_RGN_NM'] == '강남구')
]['AMT'].sum()

# 결과 출력
print("강남구 편의점 총 소비액:", f"{gangnam_convenience_spending:,}원")

# 조건에 맞는 데이터 필터링 (편의점 & 강남구)
gangnam_convenience_spending = df[
    (df['TP_BUZ_NM'] == '편 의 점') & (df['CTY_RGN_NM'] == '강남구')
]['AMT'].sum()

# 결과 출력
print("강남구 편의점 총 소비액:", f"{gangnam_convenience_spending:,}원")

# 내 거주 조건에 맞는 데이터 필터링 (편의점 & 관악구)
gangnam_convenience_spending = df[
    (df['TP_BUZ_NM'] == '편 의 점') & (df['CTY_RGN_NM'] == '관악구')
]['AMT'].sum()

# 결과 출력
print("관악구 편의점 총 소비액:", f"{gangnam_convenience_spending:,}원")




























