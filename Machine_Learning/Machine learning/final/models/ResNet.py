import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    '''
      Этот класс должен быть реализован аналогично BasicBlock из имплементации pytorch.
    
      Поскольку в resnet18 все блоки имеют только два параметра, отвечающие за число каналов, 
    мы также будем использовать упрощённую нотацию. 

      Первый свёрточный слой имеет размерность (in_channels,  out_channels), 
    второй свёрточный слой имеет размерность (out_channels, out_channels).

      От вас требуется реализовать правильный forward и init для этого блока.
    
      Помните, что необходимо использовать batch-нормализации и функции активации. 
      Shorcut потребует от вас понимания, что такое projection convolution. 

      В общем и целом, рекомендуется обратиться к оригинальной статье, имплементации pytorch и 
    другими источникам информации, чтобы успешно собрать этот блок. 

    Hint! Вам может помочь nn.Identity() для реализации shorcut.
    '''

    def __init__(self, in_channels, out_channels):    
        '''
        У блока должны быть объявлены следующие поля:
            * self.shorcut 
            * self.activation
            * self.conv1
            * self.conv2
            * self.bn1
            * self.bn2

        Hint! Не забывайте про параметры bias, padding и stride у свёрточных слоёв.
        '''
        
        super().__init__()
        stride = (2, 2) if in_channels != out_channels else (1, 1)
        
        # <----- your code here ----->
        raise NotImplementedError()
        

    def forward(self, x):
        '''
        '''

        residual = self.shortcut(x)
        
        # <----- your code here ----->
        raise NotImplementedError()
        
        return x + residual


class ResNetLayer(nn.Module):
    '''
    Этот класс должен быть реализован аналогично layer из имплементации pytorch.
    
    Для реализации слоя потребуется создать внутри два ResidualBlock. 
    Определение соответствующих размерностей остаётся за вами. 

    '''

    def __init__(self, in_channels, out_channels):
        '''
        У слоя должно быть объявлено следующее поле:
            * self.blocks
        '''

        super().__init__()
        
        # <----- your code here ----->
        raise NotImplementedError()
        
    def forward(self, x):
        '''
          Обратите внимание, что blocks должен быть запакован так, 
        чтобы forward работал в исходном виде.
        '''
        
        x = self.blocks(x)
        return x


class ResNet18(nn.Module):
    '''
    Наконец, этот класс должен состоять из трёх основных компонентов:
      1. Четырёх подготовительных слоёв.
      2. Набора внутренних ResNetLayer.  
      3. Финального классификатора. 

    Hint! Чтобы сеть могла обрабатывать изображения из CIFAR10, следует заменить параметры
          первого свёрточного слоя на kernel_size=(3, 3), stride=(1, 1) и padding=(1, 1).
    '''

    def __init__(self, in_channels=3, n_classes=10):
        '''
        У класса должны быть объявлены следующие поля:
            * self.conv1
            * self.bn1
            * self.activation
            * self.maxpool
            * self.layers
            * self.avgpool
            * self.flatten
            * self.fc

        Допускается иная группировка параметров, не нарушающая смысла архитектуры сети.
        '''

        super().__init__()
        
        # <----- your code here ----->
        raise NotImplementedError()
        
    def forward(self, x):


        # <----- your code here ----->
        raise NotImplementedError()
        
        return x