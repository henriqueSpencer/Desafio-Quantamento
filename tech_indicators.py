#Import libraries to compute data
import numpy as np
import math

class TI:
    #values should be in this order inside an array = [close_values,open_values,high_values,low_values,volume]
    def __init__(self,values):
        self.values = values

    #Simple Moving Average:
    #receives historic 1D array = values
    #returns a 2D array: SMA = [[SMA], [std_dev]]
    def SMA(self,time_period,column=0):
        values = self.values[column]      
        sma = [values[0]]
        std_dev = [0]
        for i in range(1,len(values)):
            if i < time_period:
                sma.append(np.mean(values[0:i]))
                std_dev.append(np.std(values[0:i]))
            else:
                sma.append(np.mean(values[i-time_period:i]))
                std_dev.append(np.std(values[i-time_period:i]))
        return np.array([sma, std_dev])

    #Exponential Moving Avarage:
    #receives historic 1D array = values, 
    #returns a 1D array: [EMA_values]
    def EMA(self,time_period,column=0):
        values = self.values[column]
        k = 2 / (1 + time_period)
        sma = np.mean(values[:math.ceil(time_period*0.1)])  
        ema_values = [sma]      
        for i in range(0,len(values)):
            ema_values.append(values[i] * k + ema_values[-1] * (1 - k))
        ema_values.pop(0)
        return np.array(ema_values)

    #Average True Range for volatility
    #receives historic 2D array = values = [close_values, max_values, min_values]
    #returns a 2D array: [[SMA], [standard deviation]]
    def ATR(self,time_period):
        values = [self.values[0],self.values[2],self.values[3]]
        atr = [(values[1][0] - values[2][0])/time_period]
        tr_1 = np.subtract(values[1],values[0])     #Cur_Max - Cur_Low
        prev_close = np.delete(np.hstack(([0],values[0])),-1)
        tr_2 = np.subtract(values[1],prev_close)       #Cur_Max - Prev_Close 
        tr_3 = np.subtract(values[2],prev_close)       #Cur_Low - Prev_Close
        tr = [abs(tr_1),abs(tr_2),abs(tr_3)] 
        for i in range(1,len(values[0])): 
            tri = [tr[0][i], tr[1][i], tr[2][i]]
            max_tr = tri.index(max(tri))
            if max_tr == 0:
                tr_i = tr_1[i]
            elif max_tr == 1:
                tr_i = tr_2[i]
            else:
                tr_i = tr_3[i]
            atr.append((atr[i-1] * (time_period-1) + tr_i)/time_period)    
        return np.array(atr)

    #Relative Strength Index
    #Formula:	RSI = 100 - 100/(1 + RS)
    #		where: RS = Average Up movements/ Average Down movements
    #              values = close_values
    def RSI (self,time_period,column=0):
        values = self.values[column]
        rsi = [0]
        up_mov = [0]
        down_mov = [0]

        for i in range(1, len(values)):
            value = values[i] - values[i-1]
            up = 0
            down = 0

            if value > 0:
                up = value 
            elif value < 0:
                down = abs(value)

            up_mov.append(up)
            down_mov.append(down)
            
            if i < time_period:
                if sum(down_mov) == 0:
                    rsi_i = 100
                else:
                    rs = sum(up_mov)/sum(down_mov)
                    rsi_i = 100 - 100/(1 + rs)
            else:
                if sum(down_mov[-time_period:]) == 0:
                    rsi_i = 100
                else:
                    rs = sum(up_mov[-time_period:])/sum(down_mov[-time_period:])				
                    rsi_i = 100 - 100/(1 + rs)
            rsi.append(rsi_i)
        rsi = np.array(rsi)
        return rsi

    #Heikin-Ashi to eliminate the noise
    #receives historic 2D array = values = [open_values, close_values, max_values, min_values]
    #returns a 2D array: HA = [[HA_open], [HA_close], [HA_max], [HA_min]]
    def HA (self,ha_type):
        values = [self.values[1],self.values[0],self.values[2],self.values[3]]
        if ha_type == "hist":
            ha_open = [values[0][0]]
            ha_close = [values[1][0]]
            ha_max = [values[2][0]]
            ha_min = [values[3][0]]
            for i in range(1, len(values[0])):
                ha_open.append(np.mean([ha_open[i-1], ha_close[i-1]]))
                ha_close.append(np.mean([values[0][i], values[1][i], values[2][i], values[3][i]]))
                ha_max.append(max(values[2][i],ha_open[i],ha_close[i]))
                ha_min.append(min(values[3][i],ha_open[i],ha_close[i]))
            ha_open = np.array(ha_open)
            ha_close = np.array(ha_close)
            ha_max = np.array(ha_max)
            ha_min = np.array(ha_min)
        elif ha_type == "next":
            ha_open = np.hstack(([values[1][0]], np.mean([values[1][0][-1], values[1][1][-1]])))
            ha_close = np.hstack(([values[1][1]], np.mean([values[0][0][-1],values[0][1][-1],values[0][2][-1],values[0][3][-1]])))
            ha_max = np.hstack((values[1][2], max(values[0][2][-1],values[1][0][-1],values[1][1][-1])))
            ha_min = np.hstack((values[1][3], min(values[0][3][-1],values[1][0][-1],values[1][1][-1])))
        return ha_close,ha_open,ha_max,ha_min

    #Keltner Channels for volatility
    def KC(self,sma, atr, factor=3):
        distance = abs(np.multiply(atr,factor))
        upper_band = np.add(sma,distance)
        lower_band = np.subtract(sma,distance)
        return upper_band,lower_band

    #Bollinger Bands for volatility
    #The sma argument contains [[SMA]]
    def BBands(self,sma, factor=2):
        distance = np.multiply(sma[1],factor)
        upper_band = np.add(sma[0],distance)
        lower_band = np.subtract(sma[0],distance)
        return upper_band,lower_band

    #Moving Average Convergence Divergence for tendence
    #It takes a short and a long period Exponential Moving Average as arguments
    def MACD(self,ema_short,ema_long):
        macd = np.subtract(ema_short,ema_long)
        return macd

    #Stochastic Oscillator
    def SO(self, time_period):
        c = self.values[0]
        h = self.values[2]
        l = self.values[3]
        k_0 = 100*(c[0] - l[0])/(h[0] - l[0])
        k = [k_0]
        for i in range(1,len(c)):
            if i < time_period:
                highest = np.amax(h[0:i])
                lowest = np.amin(l[0:i])
            else:
                highest = np.amax(h[i-time_period:i])
                lowest = np.amin(l[i-time_period:i])
            k_i = 100*(c[i] - lowest)/(highest - lowest)
            k.append(k_i)
        return np.array(k)

    #Ease Of Movement
    def EOM(self):
        h = self.values[2]
        l = self.values[3]
        v = self.values[4]
        prev_h = np.hstack((h.copy()[0],h.copy()))
        prev_h = prev_h[:-1]
        prev_l = np.hstack((l.copy()[0],l.copy()))
        prev_l = prev_l[:-1]
        distance = ((h+l)/2) - ((prev_h+prev_l)/2)
        norm_v = v/np.mean(v)
        normal_v = v/norm_v
        box_ratio = normal_v/(h-l)
        eom = distance/box_ratio
        return eom

    #Ichimoku Cloud Support and Resistance indicator
    def IchimokuCloud(self):
        c = self.values[0]
        h = self.values[2]
        l = self.values[3]
        chikou_span = [c[0]]
        cloud = [0]
        for i in range(1,len(h)):
            if i < 9:
                ph_9_i = max(h[0:i])
                pl_9_i = min(l[0:i])
            else:
                ph_9_i = max(h[i-9:i])
                pl_9_i = min(l[i-9:i])
            if i < 26:
                ph_26_i = max(h[0:i])
                pl_26_i = min(l[0:i])
                chikou_i = c[0]
            else:
                ph_26_i = max(h[i-26:i])
                pl_26_i = min(l[i-26:i])
                chikou_i = c[i-26]
            if i < 52:
                ph_52_i = max(h[0:i])
                pl_52_i = min(l[0:i])
            else:
                ph_52_i = max(h[i-52:i])
                pl_52_i = min(l[i-52:i])

            conversion_i = (ph_9_i + pl_9_i)/2
            base_line_i = (ph_26_i + pl_26_i)/2
            lead_span_a_i = (conversion_i + base_line_i)/2
            lead_span_b_i = (ph_52_i + pl_52_i)/2
            cloud_i = lead_span_a_i - lead_span_b_i

            cloud.append(cloud_i)
            chikou_span.append(chikou_i)	

        cloud = np.array(cloud)
        chikou_span = np.array(chikou_span)
        return cloud, chikou_span

    #Momentum
    def Momentum(self,time_period):
        c = self.values[0]
        momentum = [0]
        for i in range(1,len(c)):
            if i < time_period:
                momentum_i = c[i] - c[0]
            else:
                momentum_i = c[i] - c[i-time_period]
            momentum.append(momentum_i)
        return np.array(momentum)

    def GenTechIndicators(self):
        ema3 = self.EMA(3,column=0)
        ema9 = self.EMA(9,column=0)
        ema21 = self.EMA(21,column=0)
        rsi5 = self.RSI(5,column=0)
        rsi14 = self.RSI(14,column=0)
        rsi26 = self.RSI(26)
        atr5 = self.ATR(5)
        atr14 = self.ATR(14)
        atr26 = self.ATR(26)
        vema5 = self.EMA(5,column=4)
        vema14 = self.EMA(14,column=4)
        vema26 = self.EMA(26,column=4)
        momentum5 = self.Momentum(5)
        momentum14 = self.Momentum(14)
        momentum26 = self.Momentum(26)
        so3 = self.SO(3)
        so5 = self.SO(5)
        so14 = self.SO(14)
        so26 = self.SO(26)
        eom = self.EOM()
        cloud,chikou_span = self.IchimokuCloud()

        return ema3,ema9,ema21
